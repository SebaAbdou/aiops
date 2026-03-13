#!/usr/bin/env python3
"""
STEP 2: Train Anomaly Detection Model (Offline)
- Loads training_dataset.csv
- Trains a supervised RandomForestClassifier using is_anomaly labels
- Evaluates on a held-out test split (stratified 80/20)
- Saves model.pkl, scaler.pkl, feature_list.json
- Prints precision / recall / ROC-AUC summary
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_CSV  = Path('data/training_dataset.csv')
FEATURE_JSON = Path('training/models/feature_list.json')
MODEL_PKL    = Path('training/models/anomaly_detector.pkl')
SCALER_PKL   = Path('training/models/scaler.pkl')
MODELS_DIR   = Path('training/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 2: TRAIN ANOMALY DETECTION MODEL")
print("=" * 70)

# ─── Load dataset + features ──────────────────────────────────────────────────
df = pd.read_csv(DATASET_CSV, parse_dates=['timestamp'])
with open(FEATURE_JSON) as f:
    feature_cols = json.load(f)

print(f"\nDataset: {len(df)} rows, {len(feature_cols)} features")
print(f"   Normal  : {(df['is_anomaly'] == 0).sum()}")
print(f"   Anomaly : {(df['is_anomaly'] == 1).sum()}")

X = df[feature_cols].values
y = df['is_anomaly'].values

# ─── Train / test split (stratified 80/20) ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# ─── Scale (fit on train only to avoid leakage) ───────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"\nFeatures scaled (StandardScaler)")

# ─── Train supervised RandomForest ───────────────────────────────────────────
print(f"\nTraining RandomForestClassifier (supervised, class_weight=balanced) ...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',   # handles 54% anomaly / 46% normal imbalance
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_s, y_train)

# ─── Evaluation on held-out test set ─────────────────────────────────────────
test_preds  = clf.predict(X_test_s)
test_probas = clf.predict_proba(X_test_s)[:, 1]   # P(anomaly)

print("\n" + "=" * 70)
print("EVALUATION RESULTS  (held-out 20% test set)")
print("=" * 70)
print(classification_report(y_test, test_preds, target_names=['Normal', 'Anomaly'], zero_division=0))

cm = confusion_matrix(y_test, test_preds)
print("Confusion Matrix:")
print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

try:
    auc = roc_auc_score(y_test, test_probas)
    print(f"\nROC-AUC: {auc:.3f}")
except Exception:
    pass

# ─── Feature importances (top 10) ─────────────────────────────────────────────
fi = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])[:10]
print("\nTop-10 important features:")
for name, imp in fi:
    print(f"   {imp:.4f}  {name}")

# ─── Save artefacts ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING MODEL ARTEFACTS")
print("=" * 70)

model_bundle = {
    'classifier': clf,          # supervised RF — pipeline checks for this key
    'model_type': 'supervised',
    'version': '2.0',
    'trained_on': str(pd.Timestamp.now()),
    'n_training_samples': len(X_train),
    'n_features': len(feature_cols),
    'feature_cols': feature_cols,
}

with open(MODEL_PKL, 'wb') as f:
    pickle.dump(model_bundle, f)
print(f"   Model bundle saved to {MODEL_PKL}")

with open(SCALER_PKL, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   Scaler saved to {SCALER_PKL}")

print(f"   Features saved to {FEATURE_JSON}")

print(f"\nTraining complete!")
print(f"   Samples used : {len(X_train)} (train) + {len(X_test)} (test)")
print(f"   Features used: {len(feature_cols)}")
print(f"   Model        : RandomForestClassifier (supervised, v2.0)")
print(f"\n   Next -> deploy: kubectl apply -f k8s/")
