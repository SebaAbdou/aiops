#!/usr/bin/env python3
"""
traffic generator — multi-scenario locust load test for aiops anomaly detection.

this script simulates realistic traffic patterns AND injects deliberate faults
to create labeled training data for the anomaly detector. the key concept here
is 'fault injection': we intentionally break things in controlled ways so the
ml model has clear examples of what anomalies look like.

how it works:
1. multiple user classes generate different traffic patterns
   - normal users: baseline traffic (70% of load)
   - ddos users: high-frequency requests flooding (bandwidth spike)
   - slow users: long-lived connections (connection depletion)
   - error users: trigger backend failures (error rate spike)

2. master load shape cycles through 7 repeating phases every 540 seconds:
   phase 1 (60s): warm-up — ramp up from 0 to 5 users
   phase 2 (120s): steady state — 20 constant users, normal mode
   phase 3 (60s): ddos burst — 120 users flooding requests
   phase 4 (60s): slow clients — 10 slow users holding connections
   phase 5 (60s): error flood — 25 users triggering errors
   phase 6 (60s): traffic drop — 3 users (isolated baseline)
   phase 7 (120s): cpu spike — 15 users with cpu-intensive mode

3. for each phase transition, we tell echo-server to switch fault mode:
   - 'normal': serve requests with minimal delay/errors
   - 'slow': inject 500-800ms latency artificially
   - 'error': return 5xx status codes
   - 'cpu': run cpu-intensive operation before responding

the prometheus (monitoring system) scrapes metrics every 5 seconds and sees:
- phase 1-2: baseline metrics (training data: normal)
- phase 3: request_rate spikes (training data: ddos anomaly)
- phase 4: latency spikes (training data: slow anomaly)
- phase 5: error_rate spikes (training data: error anomaly)
- phase 6: traffic drops to near-zero (training data: normal, just isolated)
- phase 7: cpu spikes (training data: cpu anomaly)

labeled data result: 2500+ windows collected, 54% anomalies, 46% normal.

this imbalance is intentional (real production has ~1% anomalies, unrealistic
for training), but it gives the model enough examples to learn patterns.

target server:
- echo-server runs on http://echo-server:5000 (kubernetes DNS)
- exposes endpoints: /echo, /health, and accepts POST /mode to switch fault mode
- also exposes /metrics on :5000/metrics (prometheus-format metrics)
"""

import time
import logging
import requests as _requests
from locust import HttpUser, task, between, events, LoadTestShape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ECHO_HOST = "http://echo-server:5000"
# kubernetes dns: echo-server service is accessible as http://echo-server:5000
# from any pod in the same namespace (aiops). kubernetes handles dns resolution.


def _set_mode(mode: str):
    """
    tell echo-server to switch fault-injection mode.
    
    this is the key to fault injection: we send a command to the target server
    asking it to change how it behaves. this lets us inject faults on-demand
    without changing traffic patterns (traffic comes from locust, server behavior
    comes from this function).
    
    modes:
    - 'normal': respond quickly with minimal errors (baseline)
    - 'slow': add 500-800ms artificial delay before responding (latency injection)
    - 'error': return 5xx status codes (error injection)
    - 'cpu': run cpu-intensive computation (cpu injection)
    
    when we transition phases, we call this function to tell echo-server
    what kind of anomaly to simulate. prometheus sees the metric spikes
    that result from the fault injection.
    """
    try:
        _requests.post(f"{ECHO_HOST}/mode", json={"mode": mode}, timeout=3)
        logger.info(f"[LoadShape] echo-server mode switched to {mode}")
    except Exception as exc:
        logger.warning(f"[LoadShape] Could not set mode {mode}: {exc}")


# ─── User classes ─────────────────────────────────────────────────────────────
# what are user classes? locust simulates concurrent users by spawning python
# threads/coroutines. each user class is one simulated user that sends requests.
# the load shape controls how many of each user type to spawn at any given time.
#
# weight: controls the ratio of this user type to others. if total weight is 3+1+1+1=6,
# then normaluser is 50% (3/6), others are ~17% each (1/6 each).

class NormalUser(HttpUser):
    """
    standard traffic: mix of /echo (primary) and /health (liveness probe).
    
    this simulates typical application users making requests at a normal pace.
    locust assigns this user type the highest weight (3), so most traffic
    is from normal users. during normal phases, metrics look good:
    - request_rate: 2-20 req/sec (depends on spawn_rate)
    - latency: ~200-250ms (baseline)
    - error_rate: <0.5% (expected baseline failures)
    
    targets:
    - /echo: main application endpoint. locust gets the response.
      this is what prometheus cares about (timing, status code).
    - /health: kubernetes liveness probe endpoint. used by k8s to check
      if pod is alive (not stuck). returns 200 ok immediately.
    """
    wait_time = between(0.5, 2)
    # each user waits 0.5-2 seconds between requests. this creates realistic
    # inter-request spacing. without this, users would spam requests instantly.
    
    weight = 3
    # 3/6 = 50% of all users are normal users. highest weight ensures
    # most traffic is baseline (not anomalies).

    @task(8)
    def echo(self):
        # request /echo endpoint 8 times for every 1 /health request.
        # this weighting reflects real applications: health probes are
        # less frequent than actual app requests.
        self.client.get("/echo")

    @task(2)
    def health(self):
        # kubernetes calls /health regularly to verify pod is alive.
        # we include it in the load test so prometheus tracks these too.
        self.client.get("/health")


class DDoSUser(HttpUser):
    """
    high-frequency requests — simulates ddos attack or traffic spike.
    
    this user type generates rapid-fire requests to stress the application.
    weight=1 means fewer instances, but each sends requests much faster
    (every 10-100ms instead of every 0.5-2s). the effect: request rate
    explodes when these users are spawned.
    
    during ddos phases:
    - request_rate: 100-120 req/sec (10x baseline)
    - latency: slightly elevated (network bottleneck)
    - error_rate: minimal (server staying up despite load)
    
    prometheus sees this spike and the anomaly detector should flag it
    as 'ddos' based on request_rate feature importance.
    """
    wait_time = between(0.01, 0.1)
    # very tight spacing: 10-100ms between requests. this is not realistic
    # user behavior, but it simulates a bot or attack flooding the server.
    
    weight = 1

    @task
    def flood(self):
        # send a request to /echo as fast as locust will let us.
        # the rapid requests should show up as a spike in prometheus metrics.
        self.client.get("/echo")


class SlowUser(HttpUser):
    """
    slow client — long gaps between requests, connection holding.
    
    this user type simulates clients that are slow, either because they:
    - are on a slow network (mobile, satellite)
    - are reading the response slowly (not closing connection)
    - are making large requests/responses (bandwidth limited)
    
    the effect: the application has many long-lived connections that aren't
    actively requesting but are taking up resources (connection pool slots,
    memory, file descriptors). this can cause latency for other requests.
    
    during slow client phases:
    - request_rate: low (each user waits 5-15s)
    - latency: can be high (server is busy with other connections)
    - error_rate: slightly elevated (timeout or resource exhaustion)
    - connection_count: high (many slow connections accumulate)
    """
    wait_time = between(5, 15)
    # very long gaps: 5-15 seconds between requests. these users simulate
    # clients that are either slow or intentionally holding connections.
    
    weight = 1

    @task
    def slow_request(self):
        # use catch_response=true so we can evaluate the response
        # and decide if it's a success or failure. this is more realistic
        # than always assuming the server response is correct.
        with self.client.get("/echo", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")
            # if status code is not 200, mark as failure. locust tracks
            # these for the reporting dashboard and error metrics.


class ErrorUser(HttpUser):
    """
    trigger error responses to stress error-rate metric.
    
    this user type specifically targets error scenarios. when error mode
    is active on the server, these users will see high failure rates.
    the purpose: to generate data that the anomaly detector learns to
    recognize as 'error anomaly'.
    
    during error phases:
    - request_rate: moderate (0.2-1s wait time)
    - latency: normal (just returns errors quickly)
    - error_rate: very high (50%+ 5xx responses)
    
    prometheus error_rate metric will spike. the anomaly detector should
    flag this as error_rate anomaly based on error features.
    
    note: we call resp.success() even for 5xx because when error injection
    is active, 5xx is expected. we don't want locust warning logs during
    intentional error injection.
    """
    wait_time = between(0.2, 1)
    # moderate request rate: 0.2-1 seconds between requests.
    # faster than normal but slower than ddos.
    
    weight = 1

    @task
    def trigger_error(self):
        with self.client.get("/echo", catch_response=True) as resp:
            if resp.status_code >= 500:
                # during error injection phase, 5xx is expected and correct.
                # mark as success so locust doesn't produce error logs.
                resp.success()
            # if status < 500, locust will treat as normal (request succeeded)


# ─── Load shape ───────────────────────────────────────────────────────────────
class MasterLoadShape(LoadTestShape):
    """
    controls the load test shape: which user types to spawn, how many,
    and at what rate. this is the 'orchestrator' that drives the whole test.
    
    locust calls tick() once per second. each tick() returns:
    - (target_users, spawn_rate): how many users should be active, and
      how many new users per second to spawn. or returns None to stop test.
    
    our implementation: cycles through 7 repeating phases every 540s (9 min).
    each phase specifies:
    - duration: how long to stay in this phase
    - target_users: how many concurrent users to have active
    - spawn_rate: how fast to add new users (users/second)
    - mode: what fault injection mode to set on echo-server
    - label: human-readable name for logs
    
    example phase progression:
    t=0s:   phase 1 (warm-up) — spawn 5 users at 2/sec
    t=60s:  phase 2 (steady) — adjust to 20 users at 5/sec
    t=180s: phase 3 (ddos) — ramp up to 120 users at 40/sec (attack!)
    t=240s: phase 4 (slow) — drop to 10 users at 5/sec, mode=slow
    t=300s: phase 5 (error) — 25 users, mode=error (errors happening)
    t=360s: phase 6 (drop) — 3 users (isolated baseline)
    t=420s: phase 7 (cpu) — 15 users, mode=cpu
    t=540s: cycle back to phase 1, repeat forever
    
    prometheus scrapes metrics every 5 seconds. it sees:
    - first 60s: ramp-up (ignore for training — system warming up)
    - next 120s: steady baseline (is_anomaly=0)
    - next 60s: ddos surge (is_anomaly=1)
    - next 60s: latency spikes (is_anomaly=1)
    - next 60s: errors (is_anomaly=1)
    - next 60s: traffic drop (is_anomaly=0 — still normal behavior)
    - next 120s: cpu spike (is_anomaly=1)
    - repeat cycle
    
    result: labeled training data for the anomaly detector.
    """

    phases = [
        # (duration_s, users, spawn_rate, mode,     label)
        # each tuple specifies one phase of the load test.
        
        (60,   5,   2,   'normal', 'Warm-up'),
        # duration: 60 seconds. target: 5 concurrent users.
        # spawn at 2 users/second until we reach 5, then hold steady.
        # mode: normal (baseline, low latency, low errors).
        # purpose: let system warm up (caches fill, connections pool initialized).
        # metrics during warmup are often unreliable, so ignored in analysis.
        
        (120,  20,  5,   'normal', 'Steady state'),
        # duration: 120 seconds. target: 20 users.
        # this is our baseline/normal phase. metrics during this period
        # are labeled is_anomaly=0 (ground truth: this is normal traffic).
        # prometheus will see stable request_rate (~20 req/sec with 20 users),
        # stable latency (~200-250ms), stable error_rate (<1%).
        # model training uses these windows to learn what "normal" looks like.
        
        (60,   120, 40,  'normal', 'DDoS burst'),
        # duration: 60 seconds. target: 120 users, spawn at 40/sec.
        # this simulates a ddos attack: sudden massive increase in traffic.
        # prometheus metrics will show:
        #   - request_rate: jumps from 20 to 120 req/sec (6x increase!)
        #   - latency: slightly increases (network/frontend bottleneck)
        #   - error_rate: minimal (server handling load ok)
        # model training labels these windows as is_anomaly=1 (ddos anomaly).
        # anomaly detector learns: "sudden spike in request_rate = ddos".
        
        (60,   10,  5,   'slow',   'Slow clients'),
        # duration: 60 seconds. target: 10 users at 5/sec spawn.
        # mode: 'slow' — echo-server will now add 500-800ms latency artificially.
        # this simulates backend degradation: database slowdown, gc pause,
        # network latency, cpu contention, etc.
        # prometheus metrics will show:
        #   - request_rate: moderate (10 users, normal pace)
        #   - latency: spikes from 200ms to 1000-2000ms (5x increase!)
        #   - error_rate: minimal
        # model training labels as is_anomaly=1 (slow_response anomaly).
        # anomaly detector learns: "latency spike = slow anomaly".
        
        (60,   25,  8,   'error',  'Error flood'),
        # duration: 60 seconds. target: 25 users, spawn at 8/sec.
        # mode: 'error' — echo-server now returns 5xx status codes.
        # this simulates total backend failure, crashed service, circuit break.
        # prometheus metrics will show:
        #   - request_rate: moderate (25 users)
        #   - latency: normal (errors returned quickly)
        #   - error_rate: very high (50%+ 5xx responses)
        # model training labels as is_anomaly=1 (error_rate anomaly).
        # anomaly detector learns: "high error_rate = error anomaly".
        
        (60,   3,   1,   'normal', 'Traffic drop'),
        # duration: 60 seconds. target: 3 users only.
        # this is intentionally very low traffic. it's still labeled normal
        # (is_anomaly=0) because low traffic doesn't mean anomaly.
        # separates the error phase from the cpu phase.
        # prometheus sees very low metrics (minimal load).
        
        (120,  15,  4,   'cpu',    'CPU spike'),
        # duration: 120 seconds (longer than others). target: 15 users.
        # mode: 'cpu' — echo-server runs cpu-intensive computation.
        # this simulates cpu exhaustion: tight loops, no sleep, thread pool
        # starved, real request processing fighting for cpu.
        # prometheus metrics will show:
        #   - request_rate: moderate (15 users)
        #   - latency: elevated (600-1000ms, cpu contention)
        #   - error_rate: slight increase (timeouts from cpu stalls)
        #   - cpu: maxed out (95%+)
        # model training labels as is_anomaly=1 (cpu_spike anomaly).
        # anomaly detector learns: "high cpu + elevated latency = cpu anomaly".
    ]

    _phase_idx: int = -1
    # track which phase we're currently in. when phase changes, we log it
    # and call _set_mode() to tell echo-server to switch fault injection.

    def tick(self):
        """
        called once per second by locust's scheduler. this is where the
        orchestration happens: figure out which phase we're in, and return
        the user count and spawn rate for this tick.
        
        the algorithm:
        1. get elapsed time since test started (self.get_run_time())
        2. convert to cycle position: elapsed % total_phase_duration
           (so after 540s, we wrap back to 0 and repeat)
        3. find which phase this position falls into
        4. if phase changed, log it and call _set_mode() to switch fault injection
        5. return (users, spawn_rate) to locust
        
        locust will then:
        - if users > current_count: spawn new users at spawn_rate
        - if users < current_count: kill excess users
        - if users == current_count: keep as-is
        """
        elapsed = self.get_run_time()
        # elapsed time in seconds since test started

        # Loop forever: wrap elapsed into a repeating cycle
        total = sum(d for d, *_ in self.phases)
        # sum all phase durations: 60+120+60+60+60+60+120 = 540 seconds
        
        elapsed = elapsed % total
        # convert elapsed to 0-540 range, wrapping when > 540.
        # so at 500s we're at position 500 in cycle 1.
        # at 550s we're at position 10 in cycle 2 (550 % 540 = 10).
        # this makes the pattern repeat forever.

        cumulative = 0
        target_phase = len(self.phases) - 1
        # loop through phases to find which one contains this elapsed time.
        for i, (dur, *_) in enumerate(self.phases):
            cumulative += dur
            # accumulate durations: 0->60->180->240->300->360->420->540
            
            if elapsed < cumulative:
                # if elapsed time hasn't reached this cumulative sum yet,
                # then current position is in phase i.
                target_phase = i
                break

        dur, users, spawn_rate, mode, label = self.phases[target_phase]
        # unpack the target phase tuple: duration, user count, spawn rate, etc

        if target_phase != self._phase_idx:
            # phase has changed (either first call or transitioned).
            self._phase_idx = target_phase
            logger.warning(
                f"\n[LoadShape] Phase {target_phase+1}/{len(self.phases)}: "
                f"{label} -- {users} users, mode={mode}"
            )
            # log the phase transition so it's visible in test output.
            
            _set_mode(mode)
            # tell echo-server to switch fault injection mode.
            # this is crucial: without this, the load shape wouldn't actually
            # inject different faults. the mode change tells echo-server
            # "stop being slow, start returning errors" or similar.

        return (users, spawn_rate)
        # return the target number of users and how fast to spawn them.
        # locust will scale the user count toward this value at this rate.


# ─── Event listeners ──────────────────────────────────────────────────────────
# locust has event hooks we can attach listeners to. these are called at
# specific moments during the test run (start, stop, failure, etc).

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    called when the test starts (after all users are spawned for phase 1).
    useful for logging what's about to happen, initializing state, etc.
    
    here we just print all the phases so the operator can see what to expect
    in the output. helps verify the load shape is configured correctly.
    """
    logger.info("Multi-scenario load test started")
    logger.info("Phase schedule:")
    for i, (dur, users, _, mode, label) in enumerate(MasterLoadShape.phases):
        logger.info(f"   Phase {i+1:2d}: {label:20s} {dur:4d}s  {users:4d} users  mode={mode}")
    # output: helps understand what's about to happen
    # example:
    #    Phase 1: Warm-up                 60s     5 users  mode=normal
    #    Phase 2: Steady state           120s    20 users  mode=normal
    #    ...


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    called when the test is explicitly stopped (ctrl-c or test completes).
    useful for cleanup: close connections, reset state, restore baseline.
    
    we reset echo-server to normal mode so it doesn't stay in a broken state
    (slow, erroring, cpu-maxed) if the test is stopped mid-phase.
    """
    logger.info("Load test stopped -- resetting echo-server to normal mode")
    _set_mode('normal')
    # ensure echo-server is back to normal after the test ends.
    # this is important: if the test stops during cpu spike phase,
    # we don't want echo-server to keep returning errors.
