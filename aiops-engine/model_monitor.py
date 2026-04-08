#!/usr/bin/env python3
"""
Model Monitor — tracks anomaly rate, feature drift, and exposes
Prometheus-compatible /metrics for scraping by the aiops-engine.

Metrics exposed:
  aiops_anomaly_rate_1m          (gauge)  – rolling 1-min anomaly %
  aiops_feature_drift_score      (gauge)  – mean z-score drift vs training baseline
  aiops_prediction_total         (counter)
  aiops_anomaly_total            (counter)
  aiops_model_version            (gauge, label: version)
"""

import time
import json
import pickle
import threading
import os
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np

# Anomaly type numeric IDs (Prometheus gauge + Grafana value mappings)
TYPE_IDS = {
    'none':          0,
    'ddos':          1,
    'slow_response': 2,
    'error_rate':    3,
    'cpu_stress':    4,
    'unknown':       5,
}
HUMAN_TYPES = {
    'none':          'Normal',
    'ddos':          'DDoS / Traffic Spike',
    'slow_response': 'Slow Response',
    'error_rate':    'Error Rate Spike',
    'cpu_stress':    'CPU Stress',
    'unknown':       'Unknown',
}

MODELS_DIR   = Path(os.getenv('MODELS_DIR', '/aiops-engine/models'))
FEATURE_JSON = MODELS_DIR / 'feature_list.json'
SCALER_PKL   = MODELS_DIR / 'scaler.pkl'

# Rolling window = 60 records (≈ 5 min at 5-s scrape interval × 12-point windows)
WINDOW = 60


class ModelMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self.predictions   : deque[int]   = deque(maxlen=WINDOW)
        self.scores        : deque[float] = deque(maxlen=WINDOW)
        self.raw_features  : deque[list]  = deque(maxlen=WINDOW)
        self.prediction_total = 0
        self.anomaly_total    = 0
        self.model_version    = "1.0"
        self.training_mean: np.ndarray | None = None
        self.training_std : np.ndarray | None = None
        # Anomaly classification state
        self.last_anomaly_type = 'none'
        self.last_anomaly_time = None
        self.last_top_features = []   # list of (name, human_label, score)
        self._load_baseline()

    def _load_baseline(self):
        """Load training mean/std from saved scaler for drift detection."""
        try:
            with open(SCALER_PKL, 'rb') as f:
                scaler = pickle.load(f)
            self.training_mean = scaler.mean_
            self.training_std  = scaler.scale_
            print("[ModelMonitor] Baseline loaded from scaler")
        except Exception as e:
            print(f"[ModelMonitor] Could not load baseline: {e}")

    def record(self, is_anomaly: int, score: float, features: list,
               anomaly_type: str = 'none', top_features: list = None):
        """Called by pipeline on every prediction."""
        with self._lock:
            self.predictions.append(is_anomaly)
            self.scores.append(score)
            self.raw_features.append(features)
            self.prediction_total += 1
            if is_anomaly:
                self.anomaly_total += 1
                self.last_anomaly_type = anomaly_type
                self.last_anomaly_time = datetime.utcnow().strftime('%H:%M:%S')
                self.last_top_features = top_features or []
            else:
                self.last_anomaly_type = 'none'

    # ── Metric accessors (thread-safe) ────────────────────────────────────────
    def anomaly_rate_1m(self) -> float:
        with self._lock:
            if not self.predictions:
                return 0.0
            return float(np.mean(list(self.predictions)))

    def feature_drift_score(self) -> float:
        """Mean absolute z-score across all features vs training baseline."""
        with self._lock:
            if not self.raw_features or self.training_mean is None:
                return 0.0
            arr = np.array(list(self.raw_features))
            z = np.abs((arr - self.training_mean) / (self.training_std + 1e-8))
            return float(z.mean())

    def avg_anomaly_score(self) -> float:
        with self._lock:
            if not self.scores:
                return 0.0
            return float(np.mean(list(self.scores)))

    def prometheus_metrics(self) -> str:
        """Return Prometheus text-format metrics string."""
        rate  = self.anomaly_rate_1m()
        drift = self.feature_drift_score()
        avg_s = self.avg_anomaly_score()

        with self._lock:
            atype     = self.last_anomaly_type
            atime     = self.last_anomaly_time or 'never'
            top_feats = list(self.last_top_features)

        lines = [
            "# HELP aiops_anomaly_rate_1m Rolling anomaly rate over last 60 windows",
            "# TYPE aiops_anomaly_rate_1m gauge",
            f"aiops_anomaly_rate_1m {rate:.6f}",
            "",
            "# HELP aiops_feature_drift_score Mean absolute z-score drift vs training baseline",
            "# TYPE aiops_feature_drift_score gauge",
            f"aiops_feature_drift_score {drift:.6f}",
            "",
            "# HELP aiops_last_anomaly_type_id Numeric type ID: 0=none 1=ddos 2=slow 3=error 4=cpu 5=unknown",
            "# TYPE aiops_last_anomaly_type_id gauge",
            f"aiops_last_anomaly_type_id {TYPE_IDS.get(atype, 5)}",
            "",
            "# HELP aiops_last_anomaly_info Info labels for last anomaly event",
            "# TYPE aiops_last_anomaly_info gauge",
            f'aiops_last_anomaly_info{{type="{atype}",human="{HUMAN_TYPES.get(atype, atype)}",when="{atime}"}} 1',
            "",
            "# HELP aiops_last_anomaly_feature_score Root-cause feature contribution scores",
            "# TYPE aiops_last_anomaly_feature_score gauge",
        ] + [
            f'aiops_last_anomaly_feature_score{{rank="{i+1}",name="{f[0]}",label="{f[1]}"}} {f[2]:.4f}'
            for i, f in enumerate(top_feats[:3])
        ] + [
            "",
            "# HELP aiops_avg_anomaly_score Rolling average raw anomaly score",
            "# TYPE aiops_avg_anomaly_score gauge",
            f"aiops_avg_anomaly_score {avg_s:.6f}",
            "",
            "# HELP aiops_prediction_total Total predictions made since startup",
            "# TYPE aiops_prediction_total counter",
            f"aiops_prediction_total {self.prediction_total}",
            "",
            "# HELP aiops_anomaly_total Total anomalies detected since startup",
            "# TYPE aiops_anomaly_total counter",
            f"aiops_anomaly_total {self.anomaly_total}",
            "",
            f'# HELP aiops_model_info Model version info',
            f'# TYPE aiops_model_info gauge',
            f'aiops_model_info{{version="{self.model_version}"}} 1',
            "",
        ]
        return "\n".join(lines)


# Singleton used by pipeline.py
monitor = ModelMonitor()
