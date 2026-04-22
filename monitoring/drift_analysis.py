"""
Analyze accumulated data to understand drift over time.
Shows how model performance degrades day by day.
"""

import subprocess
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Phase definitions
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

def analyze_drift_over_time(df):
    """Show degradation day by day"""
    df['date'] = df['timestamp'].dt.date
    
    logger.info("\n" + "="*80)
    logger.info("DRIFT ANALYSIS: Daily Performance Degradation Over 26 Days")
    logger.info("="*80)
    logger.info(f"\n{'Date':<12} {'ROC-AUC':<10} {'CPU %':<10} {'Latency ms':<12} {'Samples':<10}")
    logger.info("-"*80)
    
    results = []
    for date in sorted(df['date'].unique()):
        df_day = df[df['date'] == date]
        
        if len(df_day) < 10:
            continue  # Skip days with too few samples
        
        y_true = df_day['ground_truth']
        y_pred = df_day['is_anomaly']
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = 0.0
        
        cpu_baseline = df_day['system_cpu_usage_percent_mean'].mean()
        latency_baseline = df_day['http_p95_latency_mean'].mean()
        
        results.append({
            'date': date,
            'roc_auc': roc_auc,
            'cpu_baseline': cpu_baseline,
            'latency_baseline': latency_baseline,
            'samples': len(df_day)
        })
        
        logger.info(f"{str(date):<12} {roc_auc:<10.3f} {cpu_baseline:<10.1f} {latency_baseline:<12.0f} {len(df_day):<10}")
    
    # Show degradation trend
    if len(results) > 1:
        logger.info("\n" + "="*80)
        logger.info("DEGRADATION TREND")
        logger.info("="*80)
        
        day1_auc = results[0]['roc_auc']
        today_auc = results[-1]['roc_auc']
        day1_cpu = results[0]['cpu_baseline']
        today_cpu = results[-1]['cpu_baseline']
        days_elapsed = (results[-1]['date'] - results[0]['date']).days
        
        logger.info(f"Day 1 (Feb 27):     ROC-AUC={day1_auc:.3f}, CPU={day1_cpu:.1f}%")
        logger.info(f"Today (Mar 25):     ROC-AUC={today_auc:.3f}, CPU={today_cpu:.1f}%")
        logger.info(f"Total change:       ROC-AUC {today_auc - day1_auc:+.3f}, CPU {today_cpu - day1_cpu:+.1f}%")
        
        if days_elapsed > 0:
            degradation_per_day = (day1_auc - today_auc) / days_elapsed
            logger.info(f"Degradation rate:   {degradation_per_day:.4f} per day")
            
            if degradation_per_day > 0:
                days_to_80 = (day1_auc - 0.80) / degradation_per_day
                logger.info(f"At this rate, ROC-AUC would hit 0.80 in: {days_to_80:.0f} days")

    logger.info("="*80)
    
    return results

if __name__ == '__main__':
    logger.info("🔍 Extracting accumulated data...")
    df = extract_csv()
    if df is None:
        exit(1)
    
    logger.info(f"✓ Loaded {len(df)} windows spanning {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    df = add_ground_truth(df)
    analyze_drift_over_time(df)
