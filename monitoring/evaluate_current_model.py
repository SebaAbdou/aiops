"""
Measure current deployed model's performance on live data.

Compare model predictions (is_anomaly column in CSV)
against ground truth (phase timing labels we know).
"""

import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Phase definitions (must match traffic-gen/locustfile.py)
PHASES = [
    (60,  0, 'warm-up'),        # seconds, is_anomaly, name
    (120, 0, 'steady'),
    (60,  1, 'ddos'),
    (60,  1, 'slow'),
    (60,  1, 'error'),
    (60,  0, 'traffic_drop'),
    (120, 1, 'cpu_spike'),
]
CYCLE_SECONDS = sum(p[0] for p in PHASES)  # 540

def extract_csv():
    """Get CSV from pod"""
    result = subprocess.run([
        'kubectl', 'exec', '-n', 'aiops',
        'deployment/aiops-engine',
        '--', 'cat', '/data/cleaned_dataset.csv'
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode != 0:
        logger.error(f"Failed: {result.stderr}")
        return None
    
    df = pd.read_csv(StringIO(result.stdout))
    logger.info(f"✓ Loaded {len(df)} windows from pod")
    return df

def add_ground_truth(df):
    """Add phase-based labels"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    first_time = df['timestamp'].min()
    
    labels = []
    for ts in df['timestamp']:
        elapsed = (ts - first_time).total_seconds()
        phase_pos = elapsed % CYCLE_SECONDS
        
        cumulative = 0
        found = False
        for duration, label, name in PHASES:
            if cumulative <= phase_pos < cumulative + duration:
                labels.append(label)
                found = True
                break
            cumulative += duration
        
        if not found:
            labels.append(0)
    
    df['ground_truth'] = labels
    normal = (df['ground_truth'] == 0).sum()
    anomaly = (df['ground_truth'] == 1).sum()
    logger.info(f"✓ Added labels: {normal} normal, {anomaly} anomaly")
    return df

def evaluate(df):
    """Compare predictions to ground truth"""
    y_true = df['ground_truth'].values
    y_pred = df['is_anomaly'].values
    
    # Convert to binary if needed
    if y_pred.dtype == float:
        y_pred_proba = y_pred
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred
        y_pred_proba = y_pred.astype(float)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

def print_results(results):
    """Display results"""
    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE ON CURRENT LIVE DATA")
    logger.info("="*60)
    logger.info(f"\nAccuracy:  {results['accuracy']:.3f}")
    logger.info(f"Precision: {results['precision']:.3f}")
    logger.info(f"Recall:    {results['recall']:.3f}")
    logger.info(f"F1-Score:  {results['f1']:.3f}")
    logger.info(f"ROC-AUC:   {results['roc_auc']:.3f}")
    
    cm = results['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"           Pred_Normal | Pred_Anomaly")
    logger.info(f"Actual_Normal:  {cm[0,0]:5d}   |   {cm[0,1]:5d}")
    logger.info(f"Actual_Anomaly: {cm[1,0]:5d}   |   {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    logger.info(f"\nBreakdown:")
    logger.info(f"  True Negatives:  {tn:5d} (correct normal)")
    logger.info(f"  False Positives: {fp:5d} (false alarms)")
    logger.info(f"  False Negatives: {fn:5d} (missed anomalies)")
    logger.info(f"  True Positives:  {tp:5d} (correct anomalies)")
    
    logger.info("\n" + "="*60)
    if results['roc_auc'] >= 0.90:
        logger.info("✅ GOOD: Model performing well")
    elif results['roc_auc'] >= 0.80:
        logger.info("⚠️  DEGRADED: ROC-AUC 0.80-0.90, consider retraining")
    else:
        logger.info("❌ FAILING: ROC-AUC < 0.80, retrain needed")
    logger.info("="*60)

if __name__ == '__main__':
    df = extract_csv()
    if df is None:
        exit(1)
    
    df = add_ground_truth(df)
    results = evaluate(df)
    print_results(results)
