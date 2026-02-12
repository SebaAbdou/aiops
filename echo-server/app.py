#!/usr/bin/env python3
"""
echo server — flask app providing http metrics for prometheus + fault injection.

this is the 'target' application that the load test attacks. it serves http
requests and intentionally injects faults on demand. the traffic-gen load test
tells this server to switch modes, and the server responds with the corresponding
metric changes (latency spikes, errors, cpu). prometheus scrapes the metrics
and the anomaly detector models behavior changes.

core concept: fault injection is controlled by the /mode endpoint.
when traffic-gen calls POST /mode {"mode": "slow"}, this server starts
delaying responses. this creates a clear cause-effect chain:
  traffic-gen says "slow"  →  echo-server delays responses →  prometheus
  sees latency spike  →  anomaly detector flags it as anomaly.

endpoints:
  get  /          → health summary json (app is running?)
  get  /echo      → simulated workload. latency/errors depend on current mode.
                    this is the main endpoint: traffic-gen hammers it.
  get  /health    → kubernetes liveness probe. always 200 ok.
                    k8s uses this to verify pod is alive (not stuck).
  get  /metrics   → prometheus metrics endpoint. prometheus scrapes this
                    every 5 seconds to collect: request counts, latencies,
                    errors, modes, etc.
  post /mode      → set fault-injection mode. body: {"mode": "normal"|"slow"|"error"|"cpu"}
                    locust calls this at phase transitions to switch faults.
  get  /mode      → get current mode. useful for debugging.

fault modes explained:

  normal: baseline behavior. responds quickly with few errors.
          latency: 1-500ms (mostly <200ms).
          error_rate: <0.5% (occasional timeouts).
          cpu: minimal.
          use case: training the model on "normal" baseline.

  slow: artificially delay responses. simulates backend degradation.
        latency: 2000-8000ms (2-8 seconds!).
        error_rate: <0.5%.
        cpu: low (just sleeping).
        use case: teach model to detect latency anomalies.
                  real cause: slow database, gc pause, network delay, cpu contention.

  error: return 5xx status codes. simulates cascading failures.
         latency: normal (errors returned quickly).
         error_rate: 50%+ 5xx responses.
         cpu: low.
         use case: teach model to detect error_rate anomalies.
                  real cause: crashed microservice, database down, circuit breaker open.

  cpu: burn cpu in request handlers. simulates resource exhaustion.
       latency: 600-1000ms (elevated due to cpu contention).
       error_rate: 2-5% (some requests timeout waiting for cpu).
       cpu: maxed out (95%+).
       use case: teach model to detect cpu anomalies.
                real cause: memory leak, infinite loops, thread pool exhaustion.
"""

import time
import random
import os
import threading
import math
from flask import Flask, jsonify, request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CollectorRegistry, REGISTRY
)

app = Flask(__name__)

# ─── Prometheus metrics ────────────────────────────────────────────────────────
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)
CURRENT_MODE_GAUGE = Gauge(
    'echo_server_mode',
    'Current operating mode (0=normal,1=slow,2=error,3=cpu)',
    ['mode']
)
REQUEST_ERRORS = Counter(
    'echo_server_errors_total',
    'Total intentional error responses',
    ['status_code']
)

# ─── Request instrumentation ──────────────────────────────────────────────────
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    http_request_duration_seconds.observe(duration)
    http_requests_total.labels(
        method=request.method,
        endpoint=request.path,
        status=response.status_code
    ).inc()
    return response

# ─── Mode state ───────────────────────────────────────────────────────────────
# fault modes and current state management.
#
# modes: tuple of valid modes. traffic-gen can only request these values.
# any other mode string in a POST /mode request will be rejected.
MODES = ('normal', 'slow', 'error', 'cpu')

_state_lock = threading.Lock()
# lock to ensure thread-safe mode changes. flask handles requests in separate
# threads, so if two requests try to change mode simultaneously, we need to
# prevent race conditions. the lock ensures only one thread can change mode at a time.

_current_mode = 'normal'
# global variable holding the current fault mode. all /echo requests check this
# to decide how to behave. starts as 'normal' (baseline).

for m in MODES:
    CURRENT_MODE_GAUGE.labels(mode=m).set(1 if m == 'normal' else 0)


def get_mode() -> str:
    with _state_lock:
        return _current_mode


def set_mode(new_mode: str):
    """
    change the fault injection mode. called by POST /mode endpoint.
    
    updates:
    - the global _current_mode variable (all /echo requests read this)
    - the prometheus gauge (so prometheus tracks which mode is active)
    """
    global _current_mode
    if new_mode not in MODES:
        raise ValueError(f"Unknown mode '{new_mode}'. Valid: {MODES}")
    # reject invalid modes. prevents typos or bad requests from breaking the test.
    
    with _state_lock:
        _current_mode = new_mode
    # thread-safe update: only one thread can change mode at a time.
    
    for m in MODES:
        CURRENT_MODE_GAUGE.labels(mode=m).set(1 if m == new_mode else 0)
    # update prometheus gauge: set new_mode to 1 (active), others to 0 (inactive).
    # prometheus can then track when mode changes happen over time.


def _cpu_burn(seconds: float):
    """
    intentionally burn cpu for a specified duration.
    used in 'cpu' fault mode to simulate cpu-intensive operations.
    
    implementation: tight loop doing floating-point math (sqrt, random).
    why: math operations are cpu-bound (not i/o bound), so they actually
    consume cpu cycles (not just sleep). this increases system cpu %.
    
    effect on metrics:
    - latency: increased (request is blocked doing cpu work)
    - cpu: increased (this request consumes cpu)
    - error_rate: may increase (other requests timeout waiting for cpu)
    
    parameter: seconds — how long to burn cpu. random 0.2-0.6s in /echo mode.
    """
    end = time.time() + seconds
    x = 1.0
    while time.time() < end:
        x = math.sqrt(x + random.random())


# ─── Routes ───────────────────────────────────────────────────────────────────
# http endpoints exposed by the server. all routes report their findings to prometheus.

@app.route('/')
def index():
    # health check endpoint. returns basic info about the service.
    return jsonify({'service': 'echo-server', 'mode': get_mode(), 'status': 'ok'})


@app.route('/health')
def health():
    # kubernetes liveness probe. kubernetes calls this endpoint to check if the pod
    # is still alive. this route ALWAYS returns 200 + healthy, regardless of mode.
    # why? if we returned errors here, k8s would kill the pod thinking it crashed.
    # but we want errors only in /echo (the workload endpoint), not the health check.
    return jsonify({'status': 'healthy', 'mode': get_mode()})


@app.route('/echo', methods=['GET', 'POST'])
def echo():
    """
    main application endpoint. simulates a real workload but injects faults
    based on current mode. this is what the load test hammers (traffic-gen).
    
    the fault injection happens here: depending on the current mode, we either
    respond normally, delay, return errors, or burn cpu. prometheus metrics
    (request rate, latency, error rate, cpu) reflect these changes.
    
    flow:
    1. get current mode
    2. based on mode, inject corresponding fault
    3. record latency and status in prometheus metrics
    4. return response
    """
    mode = get_mode()
    # check what mode we're in. this read happens before prometheus metric
    # instrumentation so timing is accurate.
    
    start = time.time()
    # record when request started. used to calculate latency.

    if mode == 'normal':
        # baseline behavior. respond quickly with minimal delay.
        delay = random.uniform(0.001, 0.5)
        # 1-500ms delay. random so timing doesn't look artificial.
        # mostly <200ms (some requests faster, some slower).
        time.sleep(delay)
        status = 200
        # always succeed in normal mode.

    elif mode == 'slow':
        # latency injection: respond very slowly.
        # simulates backend degradation: slow database, gc pause, network delay.
        delay = random.uniform(2.0, 8.0)
        # 2-8 seconds delay! this is huge compared to normal (1-500ms).
        # prometheus will see latency spike 10-100x.
        time.sleep(delay)
        status = 200
        # still respond successfully (not an error, just slow).
        # anomaly detector learns: high latency = anomaly.

    elif mode == 'error':
        # error injection: return 5xx status codes.
        # simulates backend failure: service down, circuit breaker open, db crash.
        roll = random.random()
        # random 0-1 to decide which error to return.
        
        if roll < 0.5:
            # 50% chance: internal server error (500).
            time.sleep(random.uniform(0.001, 0.05))
            # errors return quickly (not slow), just fail immediately.
            REQUEST_ERRORS.labels(status_code='500').inc()
            # increment error counter for prometheus.
            return Response('Internal Server Error', status=500)
        elif roll < 0.8:
            # 30% chance: service unavailable (503).
            time.sleep(random.uniform(0.001, 0.05))
            REQUEST_ERRORS.labels(status_code='503').inc()
            return Response('Service Unavailable', status=503)
        else:
            # 20% chance: success (200). even in error mode, some requests succeed.
            # this is realistic: not 100% failure, more like 50-80% error rate.
            time.sleep(random.uniform(0.001, 0.2))
            status = 200
        # anomaly detector learns: high error_rate = anomaly.

    elif mode == 'cpu':
        # cpu injection: run cpu-intensive operation.
        # simulates resource exhaustion: memory leak, thread pool starved, loops.
        burn_time = random.uniform(0.2, 0.6)
        # burn cpu for 0.2-0.6 seconds per request.
        _cpu_burn(burn_time)
        # call the cpu-burning function. this blocks the request thread
        # and consumes cpu. other requests waiting for cpu will see latency.
        status = 200
        # request succeeds (doesn't error), just takes longer due to cpu competition.
        # anomaly detector learns: high cpu + elevated latency = anomaly.

    else:
        # fallback (shouldn't happen if mode validation is correct).
        time.sleep(random.uniform(0.001, 0.5))
        status = 200

    elapsed = time.time() - start
    return jsonify({'echo': 'ok', 'mode': mode, 'latency_ms': round(elapsed * 1000, 2)}), status


@app.route('/mode', methods=['GET'])
def get_mode_route():
    # GET /mode — query the current fault injection mode.
    # useful for debugging: which mode is active right now?
    return jsonify({'mode': get_mode()})


@app.route('/mode', methods=['POST'])
def set_mode_route():
    # POST /mode — set the fault injection mode.
    # traffic-gen calls this at phase transitions to change behavior.
    # 
    # usage: POST /mode with json body: {"mode": "normal"|"slow"|"error"|"cpu"}
    # response: {"mode": "...", "status": "ok"} or 400 error if mode invalid.
    data = request.get_json(silent=True) or {}
    new_mode = data.get('mode', '').strip()
    try:
        set_mode(new_mode)
        return jsonify({'mode': new_mode, 'status': 'ok'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/metrics', methods=['GET'])
def metrics():
    # GET /metrics — prometheus scrape endpoint.
    # prometheus calls this every 5 seconds to collect metrics from this application.
    # metrics include:
    #   - requests_total (count)
    #   - request_duration_seconds (histogram: latency distribution)
    #   - errors_total (count by status code)
    #   - current_mode_active (gauge: which mode is 1.0, others 0.0)
    #
    # these metrics are pooled across all requests and reported in prometheus exposition format.
    return Response(generate_latest(REGISTRY), mimetype='text/plain; version=0.0.4; charset=utf-8')


if __name__ == '__main__':
    # start the flask server on the configured port (default 5000, overridable via PORT env var).
    # kubernetes passes PORT=5000 via deployment.yaml.
    # 
    # threaded=True: enable threading so flask can handle multiple concurrent requests.
    # this is important because several requests will arrive simultaneously from the load test,
    # and each may block (sleeping, cpu burning) for several seconds.
    # 
    # debug=False: disable flask debug mode in production. debug mode auto-reloads code
    # on changes (not needed in k8s) and exposes the interactive debugger (security risk).
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
