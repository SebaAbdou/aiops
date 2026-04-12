#!/usr/bin/env python3
"""
MLOps Status Tracking & Prometheus Metrics Exporter

Maintains real-time status of the automated ML pipeline:
- Model performance (ROC-AUC, accuracy)
- Retraining progress (step 1/5, 2/5, etc)
- Drift detection status
- Rollback decisions
- Data collection volume

Exports metrics compatible with Prometheus scraping.
Status file: /shared/mlops_status.json (PVC)
Metrics file: /shared/mlops_metrics.txt (Prometheus format)
"""

import json
import time
import os
import math
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsStatusTracker:
    """Tracks and exports MLOps pipeline status"""
    
    def __init__(self, status_file='/shared/mlops_status.json', metrics_file='/shared/mlops_metrics.txt'):
        self.status_file = status_file
        self.metrics_file = metrics_file
        self.ensure_shared_dir()
        self.initialize_status()
    
    def ensure_shared_dir(self):
        """Create /shared directory if it doesn't exist"""
        os.makedirs('/shared', exist_ok=True)
    
    def initialize_status(self):
        """Initialize or load existing status file"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
                logger.info("Loaded existing status file")
            except:
                self.status = self._default_status()
                self.save_status()
        else:
            self.status = self._default_status()
            self.save_status()
    
    def _default_status(self):
        """Default status structure"""
        return {
            "version": "1.0",
            "initialized_at": datetime.utcnow().isoformat(),
            "last_update": datetime.utcnow().isoformat(),
            "model_health": {
                "current_roc_auc": None,
                "current_accuracy": None,
                "status": "unknown",  # healthy, degrading, critical
                "threshold_critical": 0.80,
                "threshold_warning": 0.85,
                "last_evaluation": None
            },
            "drift_detection": {
                "drift_detected": False,
                "latest_roc_auc": None,
                "samples_collected": 0,
                "samples_needed_for_retrain": 200,
                "samples_optimal": 400,
                "last_check": None
            },
            "retraining": {
                "status": "idle",  # idle, in_progress, completed, failed
                "current_step": None,  # step 1/5, 2/5, etc
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "new_model_roc_auc": None,
                "previous_model_roc_auc": None,
                "rollback_triggered": False,
                "rollback_reason": None,
                "model_version": None,
                "previous_model_version": None
            },
            "events_log": []  # Audit trail of all decisions
        }
    
    def save_status(self):
        """Write status to JSON file"""
        self.status["last_update"] = datetime.utcnow().isoformat()
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
            logger.info(f"Status saved to {self.status_file}")
            self.export_prometheus_metrics()
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def update_model_health(self, roc_auc, accuracy, evaluation_timestamp=None):
        """Update current model performance metrics"""
        roc_auc_value = round(roc_auc, 4) if isinstance(roc_auc, (int, float)) and math.isfinite(roc_auc) else None
        accuracy_value = round(accuracy, 4) if isinstance(accuracy, (int, float)) and math.isfinite(accuracy) else None

        self.status["model_health"]["current_roc_auc"] = roc_auc_value
        self.status["model_health"]["current_accuracy"] = accuracy_value
        self.status["model_health"]["last_evaluation"] = evaluation_timestamp or datetime.utcnow().isoformat()
        
        # Determine health status
        if roc_auc_value is None:
            self.status["model_health"]["status"] = "unknown"
        elif roc_auc_value < 0.80:
            self.status["model_health"]["status"] = "critical"
        elif roc_auc_value < 0.85:
            self.status["model_health"]["status"] = "degrading"
        else:
            self.status["model_health"]["status"] = "healthy"
        
        self.log_event("model_health_updated", {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "status": self.status["model_health"]["status"]
        })
        self.save_status()
    
    def update_drift_detection(self, drift_detected, roc_auc, samples_collected):
        """Update drift detection status"""
        self.status["drift_detection"]["drift_detected"] = drift_detected
        self.status["drift_detection"]["latest_roc_auc"] = round(roc_auc, 4) if roc_auc else None
        self.status["drift_detection"]["samples_collected"] = samples_collected
        self.status["drift_detection"]["last_check"] = datetime.utcnow().isoformat()
        
        action = "drift_detected" if drift_detected else "no_drift"
        self.log_event(action, {
            "roc_auc": roc_auc,
            "samples": samples_collected,
            "threshold": 0.80
        })
        self.save_status()
    
    def start_retraining(self, model_version, previous_roc_auc):
        """Mark retraining as started"""
        self.status["retraining"]["status"] = "in_progress"
        self.status["retraining"]["current_step"] = "step 1/5: Data collection"
        self.status["retraining"]["started_at"] = datetime.utcnow().isoformat()
        self.status["retraining"]["model_version"] = model_version
        self.status["retraining"]["previous_model_roc_auc"] = round(previous_roc_auc, 4) if previous_roc_auc else None
        
        self.log_event("retraining_started", {
            "model_version": model_version,
            "previous_roc_auc": previous_roc_auc
        })
        self.save_status()
    
    def update_retraining_step(self, step_number, step_name, details=None):
        """Update retraining progress"""
        self.status["retraining"]["current_step"] = f"step {step_number}/5: {step_name}"
        
        event_details = {"step": step_name}
        if details:
            event_details.update(details)
        
        self.log_event(f"retraining_step_{step_number}", event_details)
        self.save_status()
    
    def complete_retraining(self, new_roc_auc, model_version):
        """Mark retraining as completed"""
        started_at = self.status["retraining"].get("started_at")
        if isinstance(started_at, str):
            started = datetime.fromisoformat(started_at)
            duration = (datetime.utcnow() - started).total_seconds()
        else:
            duration = 0
        
        self.status["retraining"]["status"] = "completed"
        self.status["retraining"]["completed_at"] = datetime.utcnow().isoformat()
        self.status["retraining"]["duration_seconds"] = int(duration)
        self.status["retraining"]["new_model_roc_auc"] = round(new_roc_auc, 4)
        self.status["retraining"]["model_version"] = model_version
        
        self.log_event("retraining_completed", {
            "new_roc_auc": new_roc_auc,
            "previous_roc_auc": self.status["retraining"]["previous_model_roc_auc"],
            "duration_seconds": int(duration),
            "model_version": model_version
        })
        self.save_status()
    
    def trigger_rollback(self, reason, new_roc_auc, previous_roc_auc, variance_threshold=0.02):
        """Trigger model rollback if new model doesn't meet variance threshold"""
        self.status["retraining"]["status"] = "completed"
        self.status["retraining"]["rollback_triggered"] = True
        self.status["retraining"]["rollback_reason"] = reason
        
        variance = previous_roc_auc - new_roc_auc
        self.log_event("rollback_triggered", {
            "reason": reason,
            "new_roc_auc": new_roc_auc,
            "previous_roc_auc": previous_roc_auc,
            "variance": round(variance, 4),
            "threshold": variance_threshold,
            "decision": "keep_old_model"
        })
        self.save_status()
    
    def log_event(self, event_type, details):
        """Log an event to the audit trail"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "details": details
        }
        self.status["events_log"].append(event)
        logger.info(f"Event logged: {event_type} - {details}")
    
    def get_grafana_status(self):
        """Return status formatted for Grafana display"""
        health = self.status["model_health"]
        status_display = {
            "status": health["status"].upper(),
            "roc_auc": health["current_roc_auc"],
            "accuracy": health["current_accuracy"],
            "threshold_critical": health["threshold_critical"],
            "threshold_warning": health["threshold_warning"]
        }
        return status_display
    
    def export_prometheus_metrics(self):
        """Export metrics in Prometheus text format"""
        metrics = []
        h = self.status["model_health"]
        r = self.status["retraining"]
        d = self.status["drift_detection"]
        
        # Model health metrics
        if h["current_roc_auc"] is not None:
            metrics.append(f'mlops_model_roc_auc{{{self._format_labels()}}} {h["current_roc_auc"]}')
        if h["current_accuracy"] is not None:
            metrics.append(f'mlops_model_accuracy{{{self._format_labels()}}} {h["current_accuracy"]}')
        
        # Drift metrics
        metrics.append(f'mlops_drift_detected{{{self._format_labels()}}} {1 if d["drift_detected"] else 0}')
        metrics.append(f'mlops_samples_collected{{{self._format_labels()}}} {d["samples_collected"]}')
        
        # Retraining metrics
        retraining_in_progress = 1 if r["status"] == "in_progress" else 0
        metrics.append(f'mlops_retraining_in_progress{{{self._format_labels()}}} {retraining_in_progress}')
        
        if r["duration_seconds"]:
            metrics.append(f'mlops_retraining_duration_seconds{{{self._format_labels()}}} {r["duration_seconds"]}')
        
        # Write to prometheus metrics file
        try:
            with open(self.metrics_file, 'w') as f:
                f.write('\n'.join(metrics))
            logger.info(f"Prometheus metrics exported to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
    
    def _format_labels(self, job='aiops-mlops'):
        """Format Prometheus label string"""
        return f'job="{job}"'


# Convenience functions for single-line usage
_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = MLOpsStatusTracker()
    return _tracker

def save_status(status_file='/shared/mlops_status.json'):
    """Quick save status"""
    get_tracker().save_status()

def log_event(event_type, details):
    """Quick log event"""
    get_tracker().log_event(event_type, details)


if __name__ == '__main__':
    # Test functionality
    tracker = MLOpsStatusTracker()
    
    # Simulate workflow
    logger.info("=== MLOps Status Test ===")
    
    # Initial model health
    tracker.update_model_health(roc_auc=0.75, accuracy=0.73)
    
    # Detect drift
    tracker.update_drift_detection(drift_detected=True, roc_auc=0.75, samples_collected=250)
    
    # Start retraining
    tracker.start_retraining(model_version="v2.0", previous_roc_auc=0.75)
    
    # Simulate steps
    for step in range(1, 6):
        tracker.update_retraining_step(step, f"Step_{step}_Details", {"items": step})
        time.sleep(0.5)
    
    # Complete retraining
    tracker.complete_retraining(new_roc_auc=0.92, model_version="v2.0")
    
    # Export metrics
    tracker.export_prometheus_metrics()
    
    logger.info("Test completed. Check /shared/mlops_status.json")
