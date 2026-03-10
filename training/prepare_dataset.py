#!/usr/bin/env python3
"""
STEP 1: Prepare Dataset for Training
- Loads the raw collected CSV (from the running pipeline)
- Drops dead/constant features (zero variance)
- Labels anomalies by MasterLoadShape phase timing (cycles every 540s)
- Saves clean labeled training_dataset.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_CSV   = Path('data/cleaned_dataset.csv')        # live output from pipeline
OUT_CSV   = Path('data/training_dataset.csv')
FEAT_JSON = Path('training/models/feature_list.json')

# ─── MasterLoadShape cycle definition ────────────────────────────────────────
# Each tuple: (duration_s, is_anomaly, label)
# Cycle total = 540s, repeats forever
PHASES = [
    (60,  0, 'warm-up'),        # normal
    (120, 0, 'steady'),         # normal
    (60,  1, 'DDoS burst'),     # anomaly
    (60,  1, 'slow clients'),   # anomaly
    (60,  1, 'error flood'),    # anomaly
    (60,  0, 'traffic drop'),   # normal (very low traffic — still normal behaviour)
    (120, 1, 'CPU spike'),      # anomaly
]
CYCLE_S = sum(d for d, *_ in PHASES)   # 540s

def phase_label(elapsed_s: float) -> int:
    """Return 0 (normal) or 1 (anomaly) for a given elapsed second in the cycle."""
    pos = elapsed_s % CYCLE_S
    cumulative = 0
    for dur, label, _ in PHASES:
        cumulative += dur
        if pos < cumulative:
            return label
    return 0

# ─── Load ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: PREPARE DATASET")
print("=" * 70)

if not RAW_CSV.exists():
    # Fall back to dated snapshot if live CSV not present
    snapshots = sorted(Path('data').glob('cleaned_dataset_*.csv'))
    if snapshots:
        RAW_CSV = snapshots[-1]
        print(f"   Live CSV not found, using snapshot: {RAW_CSV.name}")
    else:
        raise FileNotFoundError("No dataset CSV found in data/")

df = pd.read_csv(RAW_CSV, parse_dates=['timestamp'])
print(f"   Loaded: {len(df)} rows × {len(df.columns)} cols from {RAW_CSV.name}")

if len(df) < 20:
    print(f"\nOnly {len(df)} samples — pipeline may still be warming up.")
    print(f"   Each MasterLoadShape cycle = {CYCLE_S}s (~{CYCLE_S//60} min).")
    print(f"   Wait at least {CYCLE_S*2//60} more minutes for 2 full cycles, then retry.")
    raise SystemExit(1)

# ─── Drop dead features (zero or near-zero variance) ──────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
stds = df[numeric_cols].std()
dead = stds[stds < 1e-6].index.tolist()

print(f"\nDropping {len(dead)} dead/constant features:")
for c in dead:
    print(f"   - {c}")

df.drop(columns=dead, inplace=True)

# ─── Label anomalies by load shape phase ─────────────────────────────────────
t0 = df['timestamp'].min()
df['elapsed_s'] = (df['timestamp'] - t0).dt.total_seconds()
df['is_anomaly'] = df['elapsed_s'].apply(phase_label)
df.drop(columns=['elapsed_s'], inplace=True)

n_anomaly = df['is_anomaly'].sum()
n_normal  = len(df) - n_anomaly
print(f"\nLabels assigned (by MasterLoadShape phase timing):")
print(f"   Normal    (is_anomaly=0): {n_normal}")
print(f"   Anomaly   (is_anomaly=1): {n_anomaly}")

if n_anomaly == 0:
    print("\nNo anomaly samples found. Either:")
    print("   - Pipeline hasn't reached the DDoS/error/CPU phases yet")
    print("   - Run this again after 1+ full cycles (9+ minutes of data)")

# ─── Final feature list ────────────────────────────────────────────────────────
# Exclude metadata, labels, and any model-output columns (prevent data leakage)
EXCLUDE_COLS = {'timestamp', 'is_anomaly', 'anomaly_score', 'iso_score',
                'anomaly_type', 'score', 'confidence'}
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"\nFinal feature count: {len(feature_cols)}")

# ─── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved training dataset -> {OUT_CSV}")
print(f"   {len(df)} rows x {len(df.columns)} cols")

FEAT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(FEAT_JSON, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"Saved feature list     -> {FEAT_JSON}")
print("\nDataset ready. Run: python training/train_model.py")
