#!/usr/bin/env python3
"""
Automated Retraining Pipeline with Rollback Logic

Runs as a Kubernetes Job when triggered by drift detection.

Steps:
1. Collect fresh training data from last 24h
2. Prepare data (features, labels, train/test split)
3. Train new model on fresh data
4. Evaluate new model vs previous model
5. Rollback decision: Deploy only if new_roc_auc >= (old_roc_auc - 2%)
6. Deploy new model or keep old (with detailed logging)

All steps tracked in mlops_status.json for Grafana visualization.
"""

import os
import sys
import json
import subprocess
import shutil
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, '/aiops-engine')

try:
    from mlops_status_exporter import MLOpsStatusTracker
except ImportError:
    logger.warning("mlops_status_exporter not available")
    MLOpsStatusTracker = None


class AutoRetrainingPipeline:
    """Automated ML pipeline with rollback safeguards"""
    
    def __init__(self):
        self.csv_path = Path('/data/cleaned_dataset.csv')
        self.models_dir = Path(os.getenv('MODELS_DIR', '/data/models'))
        self.training_data_path = Path('/data/training_dataset_fresh.csv')
        self.status_tracker = MLOpsStatusTracker() if MLOpsStatusTracker else None
        self.variance_threshold = 0.02  # Allow 2% variance before rollback
        self.model_version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Keep this schedule aligned with traffic-gen phases.
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
    
    def run(self):
        """Execute full retraining pipeline"""
        logger.info("=== Starting Automated Retraining Pipeline ===")
        
        try:
            if self.status_tracker:
                previous_roc_auc = self._get_previous_model_performance()
                self.status_tracker.start_retraining(self.model_version, previous_roc_auc)

            # Step 1: Collect fresh data
            logger.info("Step 1/5: Collecting fresh training data...")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(1, "Data Collection", {"status": "in_progress"})
            
            df_train = self._collect_fresh_data()
            if df_train.empty:
                logger.error("Failed to collect training data")
                if self.status_tracker:
                    self.status_tracker.update_retraining_step(1, "Data Collection", {"status": "failed", "error": "no_data"})
                return False
            
            logger.info(f"Collected {len(df_train)} samples")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(1, "Data Collection", {
                    "status": "complete",
                    "samples": len(df_train)
                })
            
            # Step 2: Prepare data
            logger.info("Step 2/5: Preparing features and labels...")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(2, "Data Preparation")
            
            df_train = self._add_ground_truth_labels(df_train)
            X_train, X_test, y_train, y_test, feature_cols = self._prepare_features(df_train)
            
            logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
            logger.info(f"Train class balance: {(y_train == 0).sum()} normal, {(y_train == 1).sum()} anomalies")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(2, "Data Preparation", {
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "normal": int((y_train == 0).sum()),
                    "anomalies": int((y_train == 1).sum())
                })
            
            # Step 3: Train new model
            logger.info("Step 3/5: Training new model...")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(3, "Model Training")
            
            train_result = self._train_model(X_train, y_train)
            new_model, new_scaler = train_result if train_result else (None, None)
            if new_model is None:
                logger.error("Model training failed")
                if self.status_tracker:
                    self.status_tracker.update_retraining_step(3, "Model Training", {"status": "failed"})
                return False
            
            logger.info("✓ Model trained successfully")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(3, "Model Training", {"status": "complete"})
            
            # Step 4: Evaluate new model
            logger.info("Step 4/5: Evaluating new model...")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(4, "Model Evaluation")
            
            new_roc_auc, new_accuracy = self._evaluate_model(new_model, new_scaler, X_test, y_test)
            logger.info(f"New model: ROC-AUC={new_roc_auc:.4f}, Accuracy={new_accuracy:.4f}")
            
            # Get previous model performance for comparison
            previous_roc_auc = self._get_previous_model_performance()
            logger.info(f"Previous model: ROC-AUC={previous_roc_auc:.4f}")
            
            if self.status_tracker:
                self.status_tracker.update_retraining_step(4, "Model Evaluation", {
                    "new_roc_auc": new_roc_auc,
                    "new_accuracy": new_accuracy,
                    "previous_roc_auc": previous_roc_auc
                })
            
            # Step 5: Deploy or rollback decision
            logger.info("Step 5/5: Making deployment decision...")
            if self.status_tracker:
                self.status_tracker.update_retraining_step(5, "Deployment Decision")
            
            deploy_success = self._deploy_with_rollback_check(
                new_model,
                new_scaler,
                feature_cols,
                new_roc_auc,
                previous_roc_auc,
            )
            
            if deploy_success:
                logger.info("✓ Retraining completed successfully and model deployed")
                if self.status_tracker:
                    self.status_tracker.complete_retraining(new_roc_auc, self.model_version)
                return True
            else:
                logger.warning("Retraining completed but model was rolled back")
                if self.status_tracker:
                    self.status_tracker.trigger_rollback(
                        reason="new_model_underperforming",
                        new_roc_auc=new_roc_auc,
                        previous_roc_auc=previous_roc_auc,
                        variance_threshold=self.variance_threshold
                    )
                return False
                
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            if self.status_tracker:
                self.status_tracker.log_event("retraining_error", {"error": str(e)})
            return False
    
    def _collect_fresh_data(self):
        """Collect data from last 24 hours"""
        if not self.csv_path.exists():
            logger.error(f"CSV not found at {self.csv_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                logger.warning("CSV exists but is empty")
                return df
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                df_recent = df[df['timestamp'] >= cutoff_time]
                if not df_recent.empty:
                    return df_recent

                logger.warning(
                    "No rows found in last 24h window; falling back to all available rows"
                )
                return df
            
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def _add_ground_truth_labels(self, df):
        """Add ground-truth labels based on phases or timestamp cycle."""
        if 'phase' not in df.columns:
            logger.info("No 'phase' column found, deriving labels from timestamp cycle")
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp available; cannot derive labels")
                return df

            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            if timestamps.isna().all():
                logger.warning("Invalid timestamps; cannot derive labels")
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
        
        phase_to_anomaly = {
            0: 0, 1: 0, 2: 0,  # normal
            3: 1, 4: 1, 5: 1, 6: 1  # anomaly
        }
        
        df['is_anomaly'] = df['phase'].map(phase_to_anomaly).fillna(0).astype(int)
        return df
    
    def _prepare_features(self, df):
        """Extract features and split into train/test"""
        # Exclude labels/prediction outputs to avoid target leakage.
        excluded_cols = {
            'timestamp',
            'phase',
            'is_anomaly',
            'anomaly_score',
            'iso_score',
            'anomaly_type',
        }

        feature_cols = []
        for col in df.columns:
            if col in excluded_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        if not feature_cols:
            raise ValueError("No numeric feature columns available for training")
        
        X = df[feature_cols].fillna(0)
        y = df['is_anomaly']
        
        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def _train_model(self, X_train, y_train):
        """Train RandomForest classifier"""
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self._last_train_samples = int(len(X_train))

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            logger.info("Model training complete")
            return model, scaler
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def _evaluate_model(self, model, scaler, X_test, y_test):
        """Evaluate model on test set"""
        try:
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            logger.info(f"Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            
            return roc_auc, accuracy
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0, 0.0
    
    def _get_previous_model_performance(self):
        """Get ROC-AUC of current deployed model from status file"""
        status_file = Path('/shared/mlops_status.json')
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                
                current_roc = status.get('model_health', {}).get('current_roc_auc')
                if current_roc is not None:
                    return current_roc
            except Exception as e:
                logger.warning(f"Could not read previous ROC-AUC: {e}")
        
        # Default to 0.5 if no previous model
        logger.info("No previous model found, assuming baseline ROC-AUC=0.5")
        return 0.5
    
    def _deploy_with_rollback_check(self, new_model, new_scaler, feature_cols, new_roc_auc, previous_roc_auc):
        """Deploy new model with rollback safeguard (variance threshold)"""
        
        variance = previous_roc_auc - new_roc_auc
        
        # Check if new model meets threshold
        if variance > self.variance_threshold and previous_roc_auc > 0.6:
            # New model is significantly worse
            logger.error(f"❌ Rollback triggered: variance {variance:.4f} exceeds threshold {self.variance_threshold}")
            logger.error(f"   New ROC-AUC {new_roc_auc:.4f} is more than {self.variance_threshold*100}% worse than previous {previous_roc_auc:.4f}")
            return False
        
        # New model is acceptable - deploy it
        logger.info(f"✓ Deployment approved: variance {variance:.4f} within threshold")
        
        try:
            model_path = self.models_dir / 'anomaly_detector.pkl'
            scaler_path = self.models_dir / 'scaler.pkl'
            feature_list_path = self.models_dir / 'feature_list.json'

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            if model_path.exists():
                backup_path = self.models_dir / f'anomaly_detector_backup_{timestamp}.pkl'
                shutil.copy(model_path, backup_path)
                logger.info(f"Backed up previous model to {backup_path}")

            model_bundle = {
                'classifier': new_model,
                'model_type': 'supervised',
                'version': self.model_version,
                'n_training_samples': int(getattr(self, '_last_train_samples', 0)),
                'n_features': int(len(feature_cols)),
                'feature_cols': feature_cols,
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_bundle, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(new_scaler, f)
            with open(feature_list_path, 'w', encoding='utf-8') as f:
                json.dump(feature_cols, f)

            # Keep backward-compatible artifact for any external scripts.
            legacy_path = self.models_dir / 'model.pkl'
            joblib.dump(new_model, legacy_path)

            logger.info(f"✓ New model deployed to {model_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False


def main():
    pipeline = AutoRetrainingPipeline()
    success = pipeline.run()
    
    exit_code = 0 if success else 1
    logger.info(f"Retraining pipeline completed with exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
