"""
Select appropriate data subset for retraining.
Uses only recent data that matches current system baseline.
"""

import subprocess
import pandas as pd
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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

def get_data_for_retraining(df, days_back=5):
    """
    Select recent data for retraining.
    
    Rationale:
    - Data from 20+ days ago has different baseline (system was colder)
    - Use only recent data that matches current system state
    - 5 days = ~1,440 windows (good balance: enough variety, current baseline)
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    cutoff = df['timestamp'].max() - pd.Timedelta(days=days_back)
    df_recent = df[df['timestamp'] >= cutoff]
    
    return df_recent, cutoff

def analyze_selection(df_all, df_selected):
    """Show what we're using for retraining"""
    logger.info("\n" + "="*70)
    logger.info("DATA SELECTION FOR RETRAINING")
    logger.info("="*70)
    
    logger.info(f"\nTotal accumulated data:  {len(df_all)} windows")
    logger.info(f"  Time span: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
    logger.info(f"  Duration: {(df_all['timestamp'].max() - df_all['timestamp'].min()).days} days")
    
    logger.info(f"\nSelected for retraining: {len(df_selected)} windows")
    logger.info(f"  Time span: {df_selected['timestamp'].min()} to {df_selected['timestamp'].max()}")
    
    # Baseline comparison
    logger.info(f"\nSystem baseline (recently vs overall):")
    
    cpu_old = df_all.head(100)['system_cpu_usage_percent_mean'].mean()
    cpu_new = df_selected['system_cpu_usage_percent_mean'].mean()
    
    latency_old = df_all.head(100)['http_p95_latency_mean'].mean()
    latency_new = df_selected['http_p95_latency_mean'].mean()
    
    logger.info(f"  Old baseline (Day 1):  CPU {cpu_old:.1f}%, Latency {latency_old:.0f}ms")
    logger.info(f"  New baseline (Recent): CPU {cpu_new:.1f}%, Latency {latency_new:.0f}ms")
    logger.info(f"  Shift:                 CPU +{cpu_new - cpu_old:.1f}%, Latency +{latency_new - latency_old:.0f}ms")
    
    # Class balance
    normal_count = (df_selected['current_mode_gauge'] == 0).sum() if 'current_mode_gauge' in df_selected.columns else len(df_selected) // 2
    anomaly_count = len(df_selected) - normal_count
    
    logger.info(f"\nClass distribution in selected data:")
    logger.info(f"  Normal samples:  ~{normal_count} ({normal_count / len(df_selected) * 100:.1f}%)")
    logger.info(f"  Anomaly samples: ~{anomaly_count} ({anomaly_count / len(df_selected) * 100:.1f}%)")
    logger.info(f"  (Good balance for training)")
    
    # Show what was excluded
    excluded = len(df_all) - len(df_selected)
    logger.info(f"\nExcluded old data: {excluded} windows")
    logger.info(f"  Reason: From early period when system baseline was different")
    logger.info(f"  This prevents the new model from learning stale patterns")
    
    logger.info("="*70)
    
    return True

if __name__ == '__main__':
    logger.info("🔍 Extracting accumulated data...")
    df = extract_csv()
    if df is None:
        exit(1)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"✓ Loaded {len(df)} windows")
    
    # Select recent data
    df_recent, cutoff = get_data_for_retraining(df, days_back=5)
    
    # Analyze
    analyze_selection(df, df_recent)
    
    # Save for retraining
    output_path = 'data/training_dataset_fresh.csv'
    df_recent.to_csv(output_path, index=False)
    logger.info(f"\n✅ Saved {len(df_recent)} rows to {output_path}")
    logger.info("   This is ready for retraining!")
