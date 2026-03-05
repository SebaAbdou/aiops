#!/usr/bin/env python3
# =============================================================================
# pipeline.py  -  the main brain of the aiops engine
# =============================================================================
#
# this file does one thing: it runs an infinite loop that watches the cluster
# and uses a machine learning model to decide if something is wrong.
#
# the full loop (repeats every 5 seconds forever):
#
#   1. query prometheus  ->  pull ~23 metrics (cpu, latency, disk, network...)
#   2. sliding window    ->  buffer the last 12 readings (= 1 minute of data)
#   3. feature extract   ->  compute 6 stats per metric = 96 features per window
#   4. data clean        ->  replace NaN / Inf with 0 so the model doesn't crash
#   5. score with model  ->  randomforest says: normal (0) or anomaly (1)?
#   6. classify type     ->  cpu_stress? slow_response? ddos? error_rate?
#   7. publish metrics   ->  expose result at :8000/metrics so prometheus scrapes it
#   8. save to csv       ->  append row to /data/cleaned_dataset.csv for retraining
#
# this file is structured as 5 classes + 1 entry point:
#   PrometheusQueryEngine  -  talks to prometheus, returns raw metric snapshots
#   SlidingWindowProcessor -  buffers last 12 snapshots, extracts 96 features
#   DataCleaner            -  sanitizes feature dict (no NaN/Inf)
#   AnomalyDetector        -  legacy unsupervised detector (kept as fallback)
#   MetricsHandler         -  tiny http server that serves /metrics for prometheus
#   AIOpsDataPipeline      -  orchestrates all of the above in one run() loop
"""
AIOps Data Pipeline:
1. Query Prometheus API every 5 seconds
2. Collect metrics from 3 layers (App, Container, Node)
3. Apply sliding window (12 points = 1 minute)
4. Clean & normalize data
5. Save cleaned dataset for ML training
"""

# ── imports ──────────────────────────────────────────────────────────────────
# requests       : send http calls to prometheus api
# pandas/numpy   : dataframes for feature matrix, math for stats
# deque          : the sliding window buffer (fixed-size, auto-drops oldest)
# pickle         : load the trained randomforest model from .pkl file
# threading      : run the /metrics http server on a background thread so the
#                  main loop isn't blocked waiting for prometheus to scrape
# sklearn        : StandardScaler (normalize features), IsolationForest + LOF
#                  (legacy unsupervised detectors, kept as fallback)
import requests
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
import json
import time
import logging
import os
import pickle
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ── model monitor ────────────────────────────────────────────────────────────
# model_monitor.py tracks live statistics: drift score, prediction counts,
# anomaly rates. it also formats everything as prometheus exposition text
# so grafana can graph them. imported as a module-level singleton (_monitor).
# if the file doesn't exist (e.g. during unit tests) we just set it to None
# and skip all monitoring calls with an 'if _monitor:' guard.
try:
    from model_monitor import monitor as _monitor
except ImportError:
    _monitor = None

# ── model file paths (read from environment variables) ───────────────────────
# these point to the model artifacts baked into the docker image.
# the kubernetes yaml sets MODELS_DIR=/app/models via the env: section.
# the Dockerfile copies them in with: COPY models/ /app/models/
#
# anomaly_detector.pkl  = trained randomforest classifier (predict normal/anomaly)
# scaler.pkl            = standardscaler (normalize the 96 features to mean=0 std=1)
# feature_list.json     = ordered list of the 96 feature names the model expects
#
# all three must exist together — if any is missing, _load_model() skips scoring
# and the pipeline falls back to data-collection-only mode.
MODELS_DIR   = Path(os.getenv('MODELS_DIR', '/aiops-engine/models'))
MODEL_DIR_CANDIDATES = [
    Path('/data/models'),
    MODELS_DIR,
    Path('/aiops-engine/models'),
]

# port where this pod exposes its own prometheus metrics (anomaly scores etc.)
# must match containerPort in 07-aiops-engine.yaml and the prometheus annotation.
METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# section 1: prometheus query engine
# =============================================================================
# responsible for: connecting to prometheus and pulling raw metric values.
#
# on startup it:
#   - waits for prometheus to be reachable (retries every 2s for up to 30 tries)
#   - probes which metric names actually exist (http? node? container?)
#   - logs a warning if any layer is missing (e.g. echo-server not running)
#
# every 5 seconds the main loop calls query_metrics() which fires ~23 promql
# queries and returns a flat dict like:
#   { 'http_request_rate': 4.2, 'system_cpu_usage_percent': 12.5, ... }
# that dict is one 'sample' = one row in the sliding window.

class PrometheusQueryEngine:
    """Query Prometheus API for metrics from all 3 layers"""
    
    def __init__(self, prometheus_url='http://prometheus:9090'):
        self.prom_url = prometheus_url
        self.data_buffer = deque(maxlen=1000)
        self.connection_retries = 0
        self.max_retries = 30
        self.available_metrics = set()  # Track which metrics are available
        
        # Wait for Prometheus to be ready
        self._wait_for_prometheus()
        self._probe_available_metrics()
    
    def _wait_for_prometheus(self):
        """Wait for Prometheus to be accessible"""
        while self.connection_retries < self.max_retries:
            try:
                response = requests.get(f'{self.prom_url}/api/v1/query', params={'query': 'up'}, timeout=2)
                if response.status_code == 200:
                    logger.info("Connected to Prometheus")
                    return
            except Exception as e:
                self.connection_retries += 1
                logger.warning(f"Waiting for Prometheus... ({self.connection_retries}/{self.max_retries})")
                time.sleep(2)
        
        logger.error("❌ Could not connect to Prometheus after retries")
        raise Exception("Prometheus not available")
    
    def _probe_available_metrics(self):
        """Check which metric types are actually available in Prometheus"""
        logger.info("🔍 Probing available metrics...")
        try:
            response = requests.get(
                f'{self.prom_url}/api/v1/label/__name__/values',
                timeout=5
            )
            if response.status_code == 200:
                self.available_metrics = set(response.json()['data'])
                
                # Categorize and report
                http_metrics = [m for m in self.available_metrics if 'http_' in m]
                node_metrics = [m for m in self.available_metrics if 'node_' in m]
                container_metrics = [m for m in self.available_metrics if 'container_' in m]
                
                logger.info(f"   📊 HTTP metrics: {len(http_metrics)} available")
                logger.info(f"   🖥️  Node metrics: {len(node_metrics)} available")
                logger.info(f"   📦 Container metrics: {len(container_metrics)} available")
                
                if len(http_metrics) == 0:
                    logger.error("   ❌ NO HTTP METRICS! Is echo-server running?")
                if len(node_metrics) == 0:
                    logger.error("   ❌ NO NODE METRICS! Is node-exporter deployed?")
                if len(container_metrics) == 0:
                    logger.warning("   No container metrics (cAdvisor may not be available)")
        except Exception as e:
            logger.warning(f"Could not probe metrics: {e}")

    
    def query_metrics(self, timestamp):
        # this is the main data collection call. it groups queries into 4 layers
        # so it's clear which part of the stack each metric comes from.
        # each layer calls _execute_query() which handles errors and returns 0.0
        # if prometheus doesn't have that metric yet (safe default).
        """Query metrics from 3 distinct layers"""
        metrics = {'timestamp': timestamp}
        
        # layer 1: application metrics (from echo-server via prometheus)
        # these come from the flask app's /metrics endpoint.
        # request_rate = how many http calls/sec are hitting the echo-server.
        # p50/p95/p99 latency = response time percentiles over the last 1 minute.
        # why percentiles? averages hide tail latency. p99=2s means 1% of users wait 2s.
        app_queries = {
            'http_request_rate': 'rate(http_requests_total[1m])',
            'http_p95_latency': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
            'http_p99_latency': 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))',
            'http_p50_latency': 'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[1m]))',
        }
        
        for metric_name, query in app_queries.items():
            metrics[metric_name] = self._execute_query(query, metric_name)
        
        # layer 2: system/os metrics (from node-exporter running on the host)
        # node-exporter is a DaemonSet that exposes the underlying VM's stats.
        # cpu is a COUNTER (always goes up), so we wrap it in rate() to get utilization.
        # memory is a GAUGE (current value), so we query it directly.
        # load_avg = number of processes waiting for cpu. >1.0 = system is busy.
        system_queries = {
            'system_cpu_usage_percent': 'rate(node_cpu_seconds_total[5m]) * 100',  # FIXED: rate() for counter
            'system_cpu_user': 'rate(node_cpu_seconds_total{mode="user"}[5m])',     # User mode CPU
            'system_cpu_system': 'rate(node_cpu_seconds_total{mode="system"}[5m])', # System mode CPU
            'system_memory_available_bytes': 'node_memory_MemAvailable_bytes / 1024',  # Convert to KB
            'system_memory_used_bytes': '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024',
            'system_load_avg_1m': 'node_load1',
            'system_load_avg_5m': 'node_load5',
        }
        
        for metric_name, query in system_queries.items():
            metrics[metric_name] = self._execute_query(query, metric_name)
        
        # layer 3: node-level i/o metrics (network + disk, from node-exporter)
        # all of these are COUNTERS so we use rate() to get bytes/sec or packets/sec.
        # without rate(): the value would just grow to billions and be useless.
        # with rate(): we get meaningful throughput numbers like '1.2 MB/s disk write'.
        node_queries = {
            'node_network_transmit_bytes_per_sec': 'rate(node_network_transmit_bytes_total[5m])',  # FIXED: rate()
            'node_network_receive_bytes_per_sec': 'rate(node_network_receive_bytes_total[5m])',    # FIXED: rate()
            'node_network_transmit_packets_per_sec': 'rate(node_network_transmit_packets_total[5m])',
            'node_network_receive_packets_per_sec': 'rate(node_network_receive_packets_total[5m])',
            'node_disk_read_bytes_per_sec': 'rate(node_disk_read_bytes_total[5m])',    # FIXED: rate()
            'node_disk_write_bytes_per_sec': 'rate(node_disk_written_bytes_total[5m])',  # FIXED: rate()
            'node_disk_io_time_ms': 'rate(node_disk_io_time_ms_total[5m])',
        }
        
        for metric_name, query in node_queries.items():
            metrics[metric_name] = self._execute_query(query, metric_name)
        
        # layer 4: container metrics (from cadvisor, built into the kubelet)
        # cadvisor tracks resource usage per-container (not per-node).
        # we sum across all pods to get cluster-wide container resource usage.
        # these may return 0 if cadvisor isn't available in this cluster setup.
        container_queries = {
            'container_cpu_usage_percent': 'sum(rate(container_cpu_usage_seconds_total{pod_name!="",pod_name!="POD"}[5m])) * 100',
            'container_memory_working_set_bytes': 'sum(container_memory_working_set_bytes{pod_name!="",pod_name!="POD"})',
            'container_network_receive_bytes_per_sec': 'sum(rate(container_network_receive_bytes_total{pod_name!="",pod_name!="POD"}[5m]))',
            'container_network_transmit_bytes_per_sec': 'sum(rate(container_network_transmit_bytes_total{pod_name!="",pod_name!="POD"}[5m]))',
            'container_fs_usage_bytes': 'sum(container_fs_usage_bytes{pod_name!="",pod_name!="POD"})',
        }
        
        for metric_name, query in container_queries.items():
            metrics[metric_name] = self._execute_query(query, metric_name)
        
        self.data_buffer.append(metrics)
        return metrics
    
    def _execute_query(self, query, metric_name):
        """Execute single PromQL query with error handling and validation"""
        try:
            response = requests.get(
                f'{self.prom_url}/api/v1/query',
                params={'query': query},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()['data']['result']
                
                if data:
                    # Successfully retrieved metric
                    value = float(data[0]['value'][1])
                    
                    # Validate value is not NaN/Inf but allow zero (legit for rates)
                    if np.isnan(value) or np.isinf(value):
                        logger.warning(f"Invalid value for {metric_name}: {value}, using 0.0")
                        return 0.0
                    
                    return value
                else:
                    # Query returned no results (metric may not exist yet)
                    logger.debug(f"No data for {metric_name} (query: {query}) - metric may not exist yet")
                    return 0.0
            else:
                logger.warning(f"Query failed for {metric_name}: Status {response.status_code}")
                logger.debug(f"   Query: {query}")
                logger.debug(f"   Response: {response.text[:200]}")
                return 0.0
        
        except requests.exceptions.Timeout:
            logger.warning(f"Query timeout for {metric_name}: {query}")
            return 0.0
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error querying Prometheus for {metric_name}")
            return 0.0
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Parse error for {metric_name}: {str(e)}")
            return 0.0
        except Exception as e:
            logger.warning(f"Unexpected error querying {metric_name}: {str(e)}")
            return 0.0

# =============================================================================
# section 2: sliding window processor
# =============================================================================
# responsible for: turning a stream of 5-second snapshots into 1-minute features.
#
# the problem with raw snapshots: they're too noisy.
# a single cpu spike at t=0 might just be a 5-second blip — not a real issue.
# but if cpu is high for 12 consecutive readings (60 seconds), that IS a real issue.
#
# so instead of feeding each snapshot directly to the model, we buffer
# the last 12 snapshots in a deque (maxlen=12 = auto-drops oldest when full).
# once the buffer is full, we extract 6 statistics per metric:
#   mean  - average level over 1 minute
#   std   - how much it fluctuated (high std = instability)
#   min   - lowest point (useful for 'did latency ever spike?')
#   max   - highest point
#   p95   - 95th percentile (like max but less sensitive to one-off outliers)
#   trend - last_value - first_value (positive = going up, negative = going down)
#
# 16 raw metrics × 6 stats = 96 features per window.
# these 96 features are what the randomforest model actually sees.

class SlidingWindowProcessor:
    """Create windowed features (12 points = 1 minute sliding window)"""
    
    def __init__(self, window_size=12):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
    
    def process(self, metric_dict):
        """Add to window and return features if window is full"""
        self.window.append(metric_dict)
        
        if len(self.window) < self.window_size:
            return None
        
        return self._extract_features()
    
    def _extract_features(self):
        """Extract statistical features from window"""
        df = pd.DataFrame(list(self.window))
        features = {'timestamp': df['timestamp'].iloc[-1]}
        
        # Skip timestamp column
        numeric_cols = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
        
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(values) > 0:
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std()
                features[f'{col}_min'] = values.min()
                features[f'{col}_max'] = values.max()
                features[f'{col}_p95'] = values.quantile(0.95)
                features[f'{col}_trend'] = values.iloc[-1] - values.iloc[0]
        
        return features

# =============================================================================
# section 3: data cleaner
# =============================================================================
# responsible for: making sure no bad values reach the model.
#
# why do we need this? some prometheus queries return:
#   - NaN  (not a number): happens when rate() has no data yet at startup
#   - Inf  (infinity): happens with division by zero in promql
#   - extremely large values (> 1e10): can happen with counter resets
#
# sklearn models raise exceptions or produce garbage predictions on NaN/Inf.
# the cleaner replaces all of these with 0.0 (a safe neutral value).
# this way even if prometheus returns junk for one metric, the pipeline
# keeps running and the model still gets a valid 96-element vector.

class DataCleaner:
    """Clean and normalize data for ML training"""
    
    @staticmethod
    def clean(features_dict):
        """Handle NaN, infinite values, and normalization"""
        cleaned = {}
        
        for key, value in features_dict.items():
            if key == 'timestamp':
                cleaned[key] = value
            else:
                try:
                    val = float(value)
                    # Replace inf and extremely large values
                    if np.isinf(val) or np.isnan(val):
                        cleaned[key] = 0.0
                    elif abs(val) > 1e10:  # Clip extreme outliers
                        cleaned[key] = 1e10 if val > 0 else -1e10
                    else:
                        cleaned[key] = val
                except (ValueError, TypeError):
                    cleaned[key] = 0.0
        
        return cleaned

# =============================================================================
# section 4: legacy unsupervised anomaly detector  (NOT used in v2.0)
# =============================================================================
# this class was the original v1.0 detection approach.
# it uses two unsupervised algorithms that DON'T need labeled training data:
#
#   isolation forest  - builds random decision trees. points that are easy to
#                       isolate (far from others) get high anomaly scores.
#                       good at detecting global outliers.
#
#   local outlier factor (lof) - compares each point's density to its neighbors.
#                       a point in a sparse region = outlier.
#                       good at detecting local anomalies in dense data.
#
# why did we replace it? unsupervised = no labels = no anomaly types.
# it could say 'anomaly!' but not 'this is cpu_stress' or 'this is ddos'.
# v2.0 uses supervisied randomforest (trained on labeled phases from locust)
# which gives us both detection AND classification.
#
# this class is kept as a fallback: if no .pkl file is found at startup,
# the pipeline could theoretically fall back to this. in practice we always
# have the .pkl so this class never runs in the current setup.

class AnomalyDetector:
    """Real-time anomaly detection using ensemble methods"""
    
    def __init__(self, min_samples=20, contamination=0.05):
        self.min_samples = min_samples
        self.contamination = contamination
        self.features_history = []
        self.scaler = StandardScaler()
        self.iso_forest = None
        self.lof = None
        self.is_trained = False
    
    def add_sample(self, features_dict):
        """Add sample to history for training"""
        # Remove timestamp for training
        feature_values = {k: v for k, v in features_dict.items() if k != 'timestamp'}
        self.features_history.append(feature_values)
    
    def train(self):
        """Train anomaly detection models once we have enough samples"""
        if len(self.features_history) < self.min_samples:
            return False
        
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(self.features_history)
            X = df.values
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.iso_forest.fit(X_scaled)
            
            # Train Local Outlier Factor
            self.lof = LocalOutlierFactor(
                n_neighbors=min(20, len(self.features_history)//2),
                contamination=self.contamination,
                novelty=False
            )
            self.lof.fit(X_scaled)
            
            self.is_trained = True
            logger.info(f"Anomaly detection models trained on {len(self.features_history)} samples")
            return True
        except Exception as e:
            logger.warning(f"Failed to train anomaly detectors: {str(e)}")
            return False
    
    def detect(self, features_dict):
        """Detect anomalies in new sample. Returns anomaly score 0-1 (1 = anomalous)"""
        if not self.is_trained or self.iso_forest is None or self.lof is None:
            return {'is_anomaly': False, 'score': 0.0, 'confidence': 0.0, 'reason': 'Models not trained yet'}
        
        try:
            # Prepare feature vector
            feature_values = np.array([features_dict.get(k, 0.0) for k in sorted(features_dict.keys()) if k != 'timestamp']).reshape(1, -1)
            X_scaled = self.scaler.transform(feature_values)
            
            # Get anomaly scores from both models
            iso_score = self.iso_forest.score_samples(X_scaled)[0]
            iso_prediction = self.iso_forest.predict(X_scaled)[0]
            
            # Normalize isolation forest score to 0-1 range
            iso_anomaly_prob = 1.0 / (1.0 + np.exp(iso_score))
            
            lof_score = self.lof.negative_outlier_factor_[0] if hasattr(self.lof, 'negative_outlier_factor_') else -1
            lof_prediction = self.lof.predict(X_scaled)[0]
            
            # Normalize LOF score to 0-1 range
            lof_anomaly_prob = 1.0 / (1.0 + np.exp(lof_score)) if lof_score != -1 else 0.5
            
            # Ensemble vote: confidence = % of models that agree
            votes = (1 if iso_prediction == -1 else 0) + (1 if lof_prediction == -1 else 0)
            confidence = votes / 2.0  # 0.0 (both say normal) to 1.0 (both say anomaly)
            
            # Overall anomaly score (average of both models' probabilities)
            anomaly_score = (iso_anomaly_prob + lof_anomaly_prob) / 2.0
            
            # Flag as anomaly if both models agree or score is very high
            is_anomaly = votes >= 1.5 or anomaly_score > 0.7
            
            return {
                'is_anomaly': is_anomaly,
                'score': float(anomaly_score),
                'confidence': float(confidence),
                'iso_vote': int(iso_prediction == -1),
                'lof_vote': int(lof_prediction == -1),
            }
        except Exception as e:
            logger.warning(f"Anomaly detection error: {str(e)}")
            return {'is_anomaly': False, 'score': 0.0, 'confidence': 0.0, 'error': str(e)}


# =============================================================================
# section 5: prometheus metrics http server
# =============================================================================
# responsible for: exposing the pipeline's own predictions as prometheus metrics.
#
# how the data flows OUT of the pipeline:
#   pipeline.py scores a window  ->  calls _monitor.record(is_anomaly, score, ...)
#   model_monitor.py stores it   ->  updates internal counters and gauges
#   prometheus scrapes :8000/metrics  ->  calls MetricsHandler.do_GET()
#   MetricsHandler calls _monitor.prometheus_metrics()  ->  returns formatted text
#   grafana queries prometheus  ->  shows the anomaly score on the dashboard
#
# the server runs in a background thread (daemon=True) so it doesn't block
# the main loop. 'daemon=True' means: if the main process exits, this thread
# is automatically killed too (no orphaned server processes).
#
# what prometheus scrapes looks like (exposition format):
#   anomaly_score 0.87
#   is_anomaly 1
#   anomaly_type{type="cpu_stress"} 1
#   model_drift_score 0.12

class MetricsHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves Prometheus-format /metrics."""
    def do_GET(self):
        if self.path == '/metrics':
            base_metrics = _monitor.prometheus_metrics() if _monitor else '# monitor unavailable\n'
            mlops_metrics_path = Path('/shared/mlops_metrics.txt')

            if mlops_metrics_path.exists():
                try:
                    mlops_metrics = mlops_metrics_path.read_text(encoding='utf-8').strip()
                    if mlops_metrics:
                        base_metrics = f"{base_metrics.rstrip()}\n{mlops_metrics}\n"
                except Exception:
                    # Never fail the endpoint because of supplemental MLOps metrics.
                    pass

            body = base_metrics.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):  # silence access log
        pass


# =============================================================================
# section 6: main pipeline orchestrator  (the class that ties everything together)
# =============================================================================
# this is the top-level class. it creates one instance of each component and
# wires them together. the run() method IS the infinite 5-second loop.
#
# startup sequence (happens once when the pod starts):
#   1. create PrometheusQueryEngine  -> wait for prometheus to be ready
#   2. create SlidingWindowProcessor -> empty 12-slot deque
#   3. create DataCleaner            -> stateless, no setup needed
#   4. resume existing CSV           -> if /data/cleaned_dataset.csv exists,
#                                       load it so we don't lose data on restart
#   5. _load_model()                 -> load anomaly_detector.pkl + scaler.pkl
#                                       + feature_list.json from /app/models/
#   6. _start_metrics_server()       -> start background http server on port 8000
#   7. run() is called from __main__ -> begins the infinite loop

class AIOpsDataPipeline:
    """End-to-end pipeline: Ingest → Window → Clean → Infer → Save"""
    
    def __init__(self, prometheus_url='http://prometheus:9090', output_file='/data/cleaned_dataset.csv'):
        self.query_engine = PrometheusQueryEngine(prometheus_url)
        self.window_processor = SlidingWindowProcessor(window_size=12)
        self.cleaner = DataCleaner()
        self.output_file = output_file
        self.all_features = []

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load existing CSV so data survives pod restarts
        if os.path.exists(output_file):
            try:
                existing = pd.read_csv(output_file)
                self.all_features = existing.to_dict('records')
                logger.info(f"📂 Resumed from existing dataset: {len(self.all_features)} rows")
            except Exception as e:
                logger.warning(f"Could not load existing dataset: {e}")

        # Load pre-trained model if available
        self.model_bundle  = None
        self.scaler        = None
        self.feature_cols  = None
        self._active_model_dir = None
        self._active_model_mtime = None
        self._load_model()

        # Start /metrics HTTP server in background thread
        self._start_metrics_server()

    def _resolve_model_artifacts(self):
        """Return first directory that contains all required model artifacts."""
        seen = set()
        for model_dir in MODEL_DIR_CANDIDATES:
            model_dir = Path(model_dir)
            if model_dir in seen:
                continue
            seen.add(model_dir)

            model_pkl = model_dir / 'anomaly_detector.pkl'
            scaler_pkl = model_dir / 'scaler.pkl'
            feature_json = model_dir / 'feature_list.json'
            if model_pkl.exists() and scaler_pkl.exists() and feature_json.exists():
                return model_dir, model_pkl, scaler_pkl, feature_json
        return None, None, None, None

    def _load_model(self):
        # loads the three model artifacts from /app/models/ inside the container.
        # these files were copied in during 'docker build' via:
        #   COPY models/ /app/models/
        # so they are always present unless you run the image without building first.
        #
        # the model_bundle is a dict saved by train_model.py containing:
        #   { 'classifier': RandomForestClassifier, 'version': '2.0',
        #     'n_training_samples': 335, 'feature_names': [...] }
        #
        # if loading fails, self.model_bundle stays None and score_window() returns
        # {'is_anomaly': 0, 'score': 0.0} without crashing the pipeline.
        """Load trained model bundle from /app/models/ (if present.)"""
        model_dir, model_pkl, scaler_pkl, feature_json = self._resolve_model_artifacts()
        if model_pkl and scaler_pkl and feature_json:
            try:
                with open(model_pkl, 'rb') as f:
                    self.model_bundle = pickle.load(f)
                with open(scaler_pkl, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(feature_json) as f:
                    self.feature_cols = json.load(f)
                self._active_model_dir = model_dir
                self._active_model_mtime = model_pkl.stat().st_mtime
                logger.info(f"Model loaded: {self.model_bundle.get('version','?')} "
                            f"({self.model_bundle.get('n_training_samples','?')} training samples, "
                            f"{len(self.feature_cols)} features) from {model_dir}")
                if _monitor:
                    _monitor.model_version = self.model_bundle.get('version', '1.0')
                    _monitor.training_mean = self.scaler.mean_
                    _monitor.training_std  = self.scaler.scale_
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Running in data-collection mode.")
        else:
            logger.warning("No model found — running in data-collection mode (no scoring).")

    def _reload_model_if_updated(self):
        """Hot-reload model artifacts when retraining writes a newer model file."""
        model_dir, model_pkl, scaler_pkl, feature_json = self._resolve_model_artifacts()
        if not model_pkl:
            return

        try:
            model_mtime = model_pkl.stat().st_mtime
        except OSError:
            return

        needs_reload = (
            self._active_model_dir != model_dir
            or self._active_model_mtime is None
            or model_mtime > self._active_model_mtime
        )
        if needs_reload:
            logger.info(f"Detected updated model artifacts in {model_dir}; reloading model")
            self._load_model()

    def _start_metrics_server(self):
        """Start Prometheus /metrics HTTP server on METRICS_PORT."""
        try:
            server = HTTPServer(('0.0.0.0', METRICS_PORT), MetricsHandler)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            logger.info(f"📡 Prometheus metrics → http://0.0.0.0:{METRICS_PORT}/metrics")
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")

    # ── Human-readable feature name helper ───────────────────────────────────
    _FEATURE_LABELS = [
        ('http_request_rate',  'Request Rate'),
        ('http_p50_latency',   'Median Latency (p50)'),
        ('http_p95_latency',   '95th Pct Latency'),
        ('http_p99_latency',   '99th Pct Latency'),
        ('http_error_rate',    'HTTP Error Rate'),
        ('http_5xx',           '5xx Error Rate'),
        ('system_cpu',         'CPU Usage'),
        ('node_cpu',           'Node CPU'),
        ('node_load',          'System Load'),
        ('system_memory',      'Memory Usage'),
        ('node_memory',        'Node Memory'),
        ('http_active',        'Active Connections'),
    ]

    @staticmethod
    def _human_label(feature_name: str) -> str:
        for key, label in AIOpsDataPipeline._FEATURE_LABELS:
            if key in feature_name:
                stat = feature_name.rsplit('_', 1)[-1]  # mean/std/min/max/trend
                return f"{label} ({stat})"
        return feature_name.replace('_', ' ').title()

    def _classify_anomaly(self, features_dict: dict) -> tuple:
        """Return (anomaly_type, top_features) using RF importances x |z-score|."""
        if self.model_bundle is None or 'classifier' not in self.model_bundle:
            return 'unknown', []
        try:
            clf = self.model_bundle['classifier']
            vec = np.array([features_dict.get(c, 0.0) for c in self.feature_cols])
            vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]
            contribs = clf.feature_importances_ * np.abs(vec_scaled)
            top_idx  = np.argsort(contribs)[::-1][:3]
            top_feats = [
                (self.feature_cols[i],
                 self._human_label(self.feature_cols[i]),
                 float(contribs[i]))
                for i in top_idx
            ]

            # Classify from the top contributors, not just the single top feature.
            scores = {
                'ddos': 0.0,
                'slow_response': 0.0,
                'error_rate': 0.0,
                'cpu_stress': 0.0,
            }
            for raw_name, _human_name, weight in top_feats:
                name = raw_name.lower()
                if any(k in name for k in ('request_rate', 'active_conn', 'network_transmit_packets', 'network_receive_packets')):
                    scores['ddos'] += weight
                if any(k in name for k in ('p50_latency', 'p95_latency', 'p99_latency', 'latency')):
                    scores['slow_response'] += weight
                if any(k in name for k in ('error_rate', 'http_5xx', '5xx', 'error')):
                    scores['error_rate'] += weight
                if any(k in name for k in ('cpu', 'load', 'memory', 'disk', 'network_', 'io_time')):
                    scores['cpu_stress'] += weight

            best_type = max(scores, key=scores.get)
            atype = best_type if scores[best_type] > 0 else 'unknown'
            return atype, top_feats
        except Exception as e:
            logger.warning(f"classify_anomaly error: {e}")
            return 'unknown', []

    def score_window(self, features_dict: dict) -> dict:
        """Run anomaly detection on one windowed+cleaned feature dict.
        Returns a dict with is_anomaly, score, and iso_score keys.
        """
        result = {'is_anomaly': 0, 'anomaly_score': 0.0, 'iso_score': 0.0}
        if self.model_bundle is None or self.scaler is None or self.feature_cols is None:
            return result
        try:
            vec = np.array([features_dict.get(c, 0.0) for c in self.feature_cols]).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec)

            if 'classifier' in self.model_bundle:
                # ── Supervised RandomForest (v2.0) ──────────────────────────
                clf = self.model_bundle['classifier']
                is_anomaly    = int(clf.predict(vec_scaled)[0])
                anomaly_score = float(clf.predict_proba(vec_scaled)[0][1])  # P(anomaly)
            else:
                # ── Legacy unsupervised IsolationForest + LOF (v1.0) ────────
                iso = self.model_bundle['iso_forest']
                lof = self.model_bundle['lof']
                iso_pred      = iso.predict(vec_scaled)[0]
                anomaly_score = float(-iso.score_samples(vec_scaled)[0])
                lof_pred      = lof.predict(vec_scaled)[0]
                is_anomaly    = int(iso_pred == -1 and lof_pred == -1)

            result = {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'iso_score': anomaly_score,
            }

            # Classify type + root causes when anomaly detected
            anomaly_type, top_features = 'none', []
            if is_anomaly:
                anomaly_type, top_features = self._classify_anomaly(features_dict)
                logger.info(
                    f"   🔍 Type: {anomaly_type.upper()}  "
                    f"Top causes: {', '.join(f[1] for f in top_features[:3])}"
                )

            if _monitor:
                _monitor.record(
                    is_anomaly=is_anomaly,
                    score=anomaly_score,
                    features=[features_dict.get(c, 0.0) for c in self.feature_cols],
                    anomaly_type=anomaly_type,
                    top_features=top_features,
                )

        except Exception as e:
            logger.warning(f"Scoring error: {e}")
        return result
    
    def run(self, duration_seconds=600, query_interval=5):
        # =====================================================================
        # the main loop  -  this is where everything happens
        # =====================================================================
        # called from __main__ with duration_seconds=86400 (24 hours).
        # kubernetes restarts the pod if it ever exits (or fails a liveness probe).
        #
        # each iteration (every 5 seconds):
        #   step 1: query_engine.query_metrics()   -> 23 promql calls -> raw dict
        #   step 2: window_processor.process()     -> add to deque, return features
        #                                             (returns None for first 11 calls
        #                                              while the window fills up)
        #   step 3: cleaner.clean()                -> replace NaN/Inf with 0.0
        #   step 3b: score_window()                -> randomforest predict + proba
        #   step 4: append to all_features list    -> will be saved to csv
        #   step 5: every 5 windows: _save_dataset() -> write csv to /data/
        #
        # the 'if window_count % 10 == 0 or score_result[is_anomaly]' log line
        # is what you see in: kubectl logs -n aiops deployment/aiops-engine
        #   -> 'window=  40  score=0.0312'
        #   -> 'ANOMALY  ->  Type: CPU_STRESS  Top causes: CPU Usage'
        """Run pipeline for X seconds, querying Prometheus every ~5 seconds"""
        logger.info(f"🚀 AIOps Pipeline started for {duration_seconds}s (window: 12×5s = 60s)")
        start_time = time.time()
        query_count = 0
        window_count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                self._reload_model_if_updated()

                # Step 1: Query Prometheus
                current_time = datetime.now()
                metrics = self.query_engine.query_metrics(current_time)
                query_count += 1
                
                # Step 2: Sliding Window (12 data points)
                windowed_features = self.window_processor.process(metrics)
                
                if windowed_features:
                    # Step 3: Clean Data
                    cleaned_features = self.cleaner.clean(windowed_features)

                    # Step 3b: Score with pre-trained model (if available)
                    score_result = self.score_window(cleaned_features)
                    cleaned_features.update(score_result)

                    self.all_features.append(cleaned_features)
                    window_count += 1

                    anomaly_flag = 'ANOMALY' if score_result['is_anomaly'] else ''
                    if window_count % 10 == 0 or score_result['is_anomaly']:
                        logger.info(
                            f"📊 window={window_count:4d}  "
                            f"score={score_result['anomaly_score']:.4f}  {anomaly_flag}"
                        )

                    # Save incrementally every 5 windows (~5 min) so the CSV
                    # is always up to date for retraining without waiting for run end
                    if window_count % 5 == 0:
                        self._save_dataset()
                
                logger.debug(f"✓ Ingested metrics (query #{query_count})")
                
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted by user")
                break
            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}")
            
            time.sleep(query_interval)
        
        # Step 4: Save Dataset
        self._save_dataset()
        logger.info(f"✨ Pipeline complete!")
        logger.info(f"   Total queries: {query_count}")
        logger.info(f"   Windowed records: {len(self.all_features)}")
    
    def _save_dataset(self):
        # writes all collected windows to /data/cleaned_dataset.csv.
        # this file is on the PVC (persistent volume claim) so it survives
        # pod restarts. without the PVC it would be lost every time the pod crashes.
        #
        # also logs a quality report:
        #   'active features'   = columns that have changed (variance > 0)
        #   'dead metrics'      = columns that are all-zero (metric not available)
        # dead metrics are columns where prometheus returned 0.0 every single query
        # (e.g. container metrics if cadvisor isn't running). train_model.py drops
        # them automatically during prepare_dataset.py preprocessing.
        #
        # how to extract this file for retraining:
        #   kubectl exec -n aiops <pod-name> -- cat /data/cleaned_dataset.csv > data/cleaned_dataset.csv
        """Save cleaned dataset to CSV for ML training"""
        if not self.all_features:
            logger.warning("No data to save")
            return
        
        df = pd.DataFrame(self.all_features)
        df.to_csv(self.output_file, index=False)
        
        # Analyze data quality
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_variance_cols = numeric_cols[df[numeric_cols].std() == 0]
        active_cols = numeric_cols[df[numeric_cols].std() > 0]
        
        logger.info(f"\n📈 Dataset saved to: {self.output_file}")
        logger.info(f"   Records: {len(df)}")
        logger.info(f"   Total features: {len(df.columns)}")
        logger.info(f"   Active features (variance > 0): {len(active_cols)}")
        logger.info(f"   Dead metrics (all zeros): {len(zero_variance_cols)}")
        
        if len(zero_variance_cols) > 0:
            logger.warning(f"\n   Dead metrics detected:")
            for col in zero_variance_cols:
                logger.warning(f"      ❌ {col}: all values = {df[col].iloc[0]}")
        
        logger.info(f"   Memory size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        logger.info(f"\n📊 Feature breakdown:")
        logger.info(f"   HTTP metrics (working): {len([c for c in df.columns if 'http_' in c])}")
        logger.info(f"   System metrics (working): {len([c for c in df.columns if 'system_' in c])}")
        logger.info(f"   Node metrics (working): {len([c for c in df.columns if 'node_' in c])}")
        logger.info(f"   Container metrics (partial): {len([c for c in df.columns if 'container_' in c])}")
        
        logger.info(f"\n🔍 Data preview:")
        logger.info(f"\n{df[active_cols].head(3).to_string()}")
        
        logger.info(f"\nIf dead metrics detected, check:")
        logger.info(f"   1. Prometheus is scraping correct targets")
        logger.info(f"   2. Rate window (5m) has enough data points")
        logger.info(f"   3. Metric exists: curl http://prometheus:9090/api/v1/label/__name__/values")

# =============================================================================
# entry point
# =============================================================================
# when kubernetes starts the container it runs: python pipeline.py
# (defined in the Dockerfile CMD instruction).
#
# we create one AIOpsDataPipeline instance and call run() with 86400 seconds = 24h.
# in practice it runs until:
#   a) 24h elapses  -> pod exits cleanly -> kubernetes restarts it immediately
#   b) an unhandled exception crashes it -> kubernetes restarts (liveness probe)
#   c) you do 'kubectl rollout restart' -> kubernetes creates new pod, kills old one
#
# PROMETHEUS_URL and MODELS_DIR are injected by kubernetes via the env: section
# in 07-aiops-engine.yaml. if running locally for testing you'd set them manually.
if __name__ == '__main__':
    # run indefinitely (restart handled by kubernetes if needed)
    pipeline = AIOpsDataPipeline(
        prometheus_url=os.getenv('PROMETHEUS_URL', 'http://prometheus:9090'),
        output_file='/data/cleaned_dataset.csv',
    )
    pipeline.run(duration_seconds=86400)  # 24h — kubernetes restarts if it exits
