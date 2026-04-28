"""
Evaluate model on ONLY recent data (last 24 hours).
This gives honest assessment of current performance.
"""

import subprocess
import pandas as pd
from io import StringIO
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PHASES = [
    (60,  0, 'warm-up'),
    (120, 0, 'steady'),
    (60,  1, 'ddos'),
    (60,  1, 'slow'),
    (60,  1, 'error'),
    (60,  0, 'traffic_drop'),
    (120, 1, 'cpu_spike'),
]
CYCLE_SECONDS = sum(p[0] for p in PHASES)

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
    return df

def slice_recent_data(df, hours_back=24):
    """Get only last N hours of data"""
    cutoff = df['timestamp'].max() - pd.Timedelta(hours=hours_back)
    df_recent = df[df['timestamp'] >= cutoff]
    return df_recent

def evaluate(df):
    """Compare predictions to ground truth"""
    y_true = df['ground_truth'].values
    y_pred = (df['is_anomaly'].values > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, df['is_anomaly'].values)
    cm = confusion_matrix(y_true, y_pred)
    
    return {'accuracy': acc, 'roc_auc': auc, 'cm': cm}

if __name__ == '__main__':
    logger.info("🔍 Extracting accumulated data...")
    df = extract_csv()
    if df is None:
        exit(1)
    
    logger.info(f"✓ Loaded {len(df)} total windows")
    
    df = add_ground_truth(df)
    
    # Get ONLY recent 24 hours
    df_recent = slice_recent_data(df, hours_back=24)
    
    logger.info(f"✓ Sliced to {len(df_recent)} windows from last 24 hours")
    logger.info(f"  Time range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")
    
    results = evaluate(df_recent)
    
    logger.info("\n" + "="*70)
    logger.info("HONEST CURRENT PERFORMANCE (Last 24 Hours Only)")
    logger.info("="*70)
    logger.info(f"\nAccuracy:  {results['accuracy']:.3f}")
    logger.info(f"ROC-AUC:   {results['roc_auc']:.3f}")
    
    cm = results['cm']
    normal_correct = cm[0,0]
    normal_flagged = cm[0,1]
    anomaly_missed = cm[1,0]
    anomaly_caught = cm[1,1]
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"           Pred_Normal | Pred_Anomaly")
    logger.info(f"Actual_Normal:  {normal_correct:5d}   |   {normal_flagged:5d}")
    logger.info(f"Actual_Anomaly: {anomaly_missed:5d}   |   {anomaly_caught:5d}")
    
    logger.info(f"\nWhat happened:")
    logger.info(f"  ✅ Correct normal:     {normal_correct:5d} windows")
    logger.info(f"  ⚠️  False alarms:      {normal_flagged:5d} windows (false positives)")
    logger.info(f"  ⚠️  Missed anomalies:  {anomaly_missed:5d} windows (false negatives)")
    logger.info(f"  ✅ Caught anomalies:   {anomaly_caught:5d} windows")
    
    logger.info("\n" + "="*70)
    if results['roc_auc'] >= 0.90:
        logger.info("✅ GOOD: Model performing well on current data")
    elif results['roc_auc'] >= 0.80:
        logger.info("⚠️  DEGRADED: Model performance declining (ROC-AUC 0.80-0.90)")
    else:
        logger.info("❌ FAILING: Model severely underperforming (ROC-AUC < 0.80)")
        logger.info("   → RETRAIN RECOMMENDED")
    logger.info("="*70)
