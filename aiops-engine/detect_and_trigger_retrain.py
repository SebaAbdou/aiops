#!/usr/bin/env python3
"""
Continuous Drift Detection & Auto-Retrain Triggering

Runs every 5 minutes via CronJob. Evaluates model performance on last 24h of data.
If ROC-AUC < 0.80 AND last retrain was > 24h ago → trigger retraining job.

Steps:
1. Load trained model from disk
2. Extract last 24h of data from CSV (in PVC)
3. Add ground-truth labels based on phases
4. Evaluate model performance
5. Check: Is drift detected? Is data ready? Is frequency OK?
6. If all conditions met → launch retraining job via Kubernetes API
7. Update mlops_status.json with findings

Runs in pod with: kubectl, python, pandas, scikit-learn access
"""

import os
import sys
import json
import subprocess
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import mlops_status_exporter
sys.path.insert(0, '/aiops-engine')

try:
    from mlops_status_exporter import MLOpsStatusTracker
except ImportError:
    logger.warning("mlops_status_exporter not available, continuing without status tracking")
    MLOpsStatusTracker = None

class DriftDetector:
    """Detects model drift and triggers retraining"""
    
    def __init__(self):
        self.csv_path = Path('/data/cleaned_dataset.csv')
        self.model_candidates = [
            Path('/data/models/anomaly_detector.pkl'),
            Path('/data/models/model.pkl'),
            Path('/aiops-engine/models/anomaly_detector.pkl'),
            Path('/aiops-engine/models/model.pkl')
        ]
        self.model_path = self._resolve_model_path()
        self.status_tracker = MLOpsStatusTracker() if MLOpsStatusTracker else None
        self.drift_threshold = float(os.getenv('DRIFT_THRESHOLD', '0.80'))
        self.min_samples_for_retrain = int(os.getenv('MIN_SAMPLES_FOR_RETRAIN', '200'))
        self.min_hours_between_retrain = float(os.getenv('MIN_HOURS_BETWEEN_RETRAIN', '24'))
        # Keep in sync with traffic-gen phase schedule (seconds, anomaly_label).
        self.phase_schedule = [
            (60, 0),   # warm-up
            (120, 0),  # steady
            (60, 1),   # ddos
            (60, 1),   # slow
            (60, 1),   # error
            (60, 0),   # traffic_drop
            (120, 1),  # cpu_spike
        ]
        self.cycle_seconds = sum(duration for duration, _ in self.phase_schedule)

    def _resolve_model_path(self):
        """Prefer persistent model in PVC, fallback to image-bundled model."""
        for candidate in self.model_candidates:
            if candidate.exists():
                logger.info(f"Using model at {candidate}")
                return candidate

        # Keep backward-compatible behavior: first candidate becomes expected path.
        return self.model_candidates[0]
    
    def check_drift(self):
        """Main drift detection workflow"""
        logger.info("=== Starting Drift Detection ===")
        
        try:
            # Step 1: Load model
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}; skipping drift evaluation this cycle")
                if self.status_tracker:
                    self.status_tracker.update_drift_detection(
                        drift_detected=False,
                        roc_auc=None,
                        samples_collected=0
                    )
                return True
            
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            
            # Step 2: Extract last 24h
            df = self._extract_last_24h()
            if df.empty:
                logger.warning("No data available from last 24h")
                if self.status_tracker:
                    self.status_tracker.update_drift_detection(
                        drift_detected=False,
                        roc_auc=None,
                        samples_collected=0
                    )
                return True
            
            logger.info(f"Extracted {len(df)} samples from last 24h")
            
            # Step 3: Add ground truth labels
            df = self._add_ground_truth_labels(df)
            
            # Step 4: Evaluate model
            roc_auc, accuracy = self._evaluate_model(model, df)
            logger.info(f"Model performance: ROC-AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")
            
            # Update status tracker
            if self.status_tracker:
                self.status_tracker.update_model_health(roc_auc, accuracy)
            
            # Step 5: Check drift threshold
            drift_detected = roc_auc < self.drift_threshold
            
            if self.status_tracker:
                self.status_tracker.update_drift_detection(
                    drift_detected=drift_detected,
                    roc_auc=roc_auc,
                    samples_collected=len(df)
                )
            
            logger.info(f"Drift detected: {drift_detected} (threshold: {self.drift_threshold})")
            
            # Step 6: Check if ready to retrain
            if drift_detected:
                should_retrain, reason = self._should_retrain(len(df))
                if should_retrain:
                    logger.info("✓ Triggering retraining job")
                    self._launch_retraining_job(roc_auc)
                    return True
                else:
                    logger.info(f"Drift detected but conditions not met for retraining ({reason})")
                    return True
            else:
                logger.info("✓ Model health is good, no action needed")
                return True
                
        except Exception as e:
            logger.error(f"Error during drift detection: {e}", exc_info=True)
            if self.status_tracker:
                self.status_tracker.log_event("drift_detection_error", {"error": str(e)})
            return False
    
    def _extract_last_24h(self):
        """Extract data from last 24 hours"""
        if not self.csv_path.exists():
            logger.error(f"CSV not found at {self.csv_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"CSV loaded: {len(df)} total rows")
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                logger.info(f"Raw timestamps sample: {df['timestamp'].head(3).tolist()}")
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                logger.info(f"After parsing - NaT count: {df['timestamp'].isna().sum()}")
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                logger.info(f"Cutoff time (24h ago): {cutoff_time}")
                logger.info(f"Min timestamp in data: {df['timestamp'].min()}")
                logger.info(f"Max timestamp in data: {df['timestamp'].max()}")
                
                df_filtered = df[df['timestamp'] >= cutoff_time]
                logger.info(f"After filtering for last 24h: {len(df_filtered)} rows")
                
                return df_filtered
            
            logger.warning("No 'timestamp' column found in CSV")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _add_ground_truth_labels(self, df):
        """Add ground-truth anomaly labels based on phases"""
        if 'phase' not in df.columns:
            logger.info("No 'phase' column found, deriving labels from timestamp cycle")
            if 'timestamp' not in df.columns:
                logger.warning("No 'timestamp' column found; using all-normal fallback labels")
                df['is_anomaly'] = 0
                return df

            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            if timestamps.isna().all():
                logger.warning("Timestamps are invalid; using all-normal fallback labels")
                df['is_anomaly'] = 0
                return df

            start_time = timestamps.min()
            labels = []
            for ts in timestamps:
                if pd.isna(ts):
                    labels.append(0)
                    continue

                phase_pos = int((ts - start_time).total_seconds()) % self.cycle_seconds
                cumulative = 0
                label = 0
                for duration, phase_label in self.phase_schedule:
                    if phase_pos < cumulative + duration:
                        label = phase_label
                        break
                    cumulative += duration
                labels.append(label)

            df['is_anomaly'] = pd.Series(labels, index=df.index, dtype='int64')
            return df
        
        # Phase mapping: 0/1/2 = normal, 3/4 = anomaly, 5/6 = anomaly
        phase_to_anomaly = {
            0: 0,  # baseline
            1: 0,  # baseline
            2: 0,  # baseline
            3: 1,  # cpu_stress
            4: 1,  # ddos
            5: 1,  # memory_leak
            6: 1,  # network_latency
        }
        
        df['is_anomaly'] = df['phase'].map(phase_to_anomaly).fillna(0).astype(int)
        
        normal_count = (df['is_anomaly'] == 0).sum()
        anomaly_count = (df['is_anomaly'] == 1).sum()
        logger.info(f"Class distribution: {normal_count} normal, {anomaly_count} anomalies")
        
        return df
    
    def _evaluate_model(self, model, df):
        """Evaluate model on recent data"""
        # Handle packaged model payloads used by pipeline artifacts.
        model_obj = model
        feature_cols = None
        if isinstance(model, dict):
            model_obj = model.get('classifier', model)
            feature_cols = model.get('feature_cols')

        # Prepare features (same as training)
        if not feature_cols:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'phase', 'is_anomaly']]
        else:
            # Keep only expected columns and create missing ones as zeros.
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
        
        X = df[feature_cols].fillna(0)
        y = df['is_anomaly']

        # If a scaler artifact exists next to the model, apply it before scoring.
        scaler = None
        scaler_path = self.model_path.parent / 'scaler.pkl'
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception:
                try:
                    scaler = joblib.load(scaler_path)
                except Exception as e:
                    logger.warning(f"Could not load scaler from {scaler_path}: {e}")
        X_eval = scaler.transform(X) if scaler is not None else X
        
        # Get predictions
        try:
            y_pred_proba = model_obj.predict_proba(X_eval)[:, 1]  # Probability of anomaly
            y_pred = (y_pred_proba > 0.5).astype(int)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.0, 0.0
        
        # Calculate metrics. ROC-AUC needs both classes present.
        if len(np.unique(y)) < 2:
            logger.warning("Ground-truth labels contain a single class; ROC-AUC unavailable for this run")
            roc_auc = float('nan')
        else:
            roc_auc = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        
        return roc_auc, accuracy
    
    def _should_retrain(self, samples_collected):
        """Check if all conditions are met for retraining"""
        if samples_collected < self.min_samples_for_retrain:
            return False, f"insufficient data: {samples_collected} < {self.min_samples_for_retrain}"

        if self.min_hours_between_retrain <= 0:
            return True, "cooldown disabled"

        # Read status file
        status_file = Path('/shared/mlops_status.json')
        
        if not status_file.exists():
            logger.info("No previous status, conditions met for first retrain")
            return True, "no previous status"
        
        try:
            with open(status_file) as f:
                status = json.load(f)
            
            # Check minimum time since last retrain
            last_completed = status.get('retraining', {}).get('completed_at')
            if last_completed:
                last_time = datetime.fromisoformat(last_completed)
                hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
                
                if hours_since < self.min_hours_between_retrain:
                    logger.info(f"Last retrain was {hours_since:.1f}h ago, need {self.min_hours_between_retrain}h")
                    return False, f"cooldown active: {hours_since:.1f}h < {self.min_hours_between_retrain}h"
            
            logger.info("✓ All conditions met for retraining")
            return True, "all conditions met"
            
        except Exception as e:
            logger.error(f"Error reading status: {e}")
            return True, "status read failed; fail-open"
    
    def _launch_retraining_job(self, current_roc_auc):
        """Launch retraining job via Kubernetes"""
        try:
            # Check if kubectl is available
            subprocess.run(['kubectl', 'version', '--client'], capture_output=True, check=True)
        except:
            logger.error("kubectl not available, cannot launch job")
            return False
        
        # Create job name with timestamp
        job_name = f"aiops-retrain-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        # Build kubectl command to create job from template
        cmd = [
            'kubectl', 'create', 'job',
            job_name,
            '--from=job/aiops-retrain',
            '-n', 'aiops'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Retraining job created: {job_name}")
            
            if self.status_tracker:
                self.status_tracker.start_retraining(
                    model_version=f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    previous_roc_auc=current_roc_auc
                )
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create job: {e.stderr}")
            logger.info("Falling back to in-pod retraining execution for demo environments")

            fallback_cmd = ['python', '/aiops-engine/auto_retrain_pipeline.py']
            try:
                fallback = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
                if fallback.stdout:
                    logger.info(fallback.stdout.strip())
                if fallback.stderr:
                    logger.info(fallback.stderr.strip())
                logger.info("Fallback retraining completed successfully")
                return True
            except subprocess.CalledProcessError as fallback_err:
                logger.error(f"Fallback retraining failed: {fallback_err.stderr}")
                return False


def main():
    detector = DriftDetector()
    success = detector.check_drift()
    
    exit_code = 0 if success else 1
    logger.info(f"Drift detection completed with exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
