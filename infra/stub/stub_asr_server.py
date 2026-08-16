#!/usr/bin/env python3
"""A fake vLLM-Omni that costs nothing to run.

The entire autoscaling half of this project (M5, M6, M8) is about how the system reacts to
queue depth. None of that logic needs a GPU to be correct — it needs a server that exposes
`vllm:num_requests_waiting` and gets slower when overloaded. This is that server.

Purpose: develop and debug the KEDA ScaledObject, the Prometheus scrape config, the Argo
Rollouts analysis template, and the spike test itself on a CPU cluster, so that when the
multi-GPU box is finally running, it only ever executes a measurement. Debugging a YAML
indentation error at $25/hour is a bad way to spend money.

It deliberately models the ONE queueing behaviour that matters: a fixed number of batch
slots, and requests that wait when all slots are full. That is what makes queue depth a
leading indicator — it starts climbing the instant arrival rate exceeds service rate, well
before latency visibly degrades.

Stdlib only, so it runs anywhere without a build step:

    python infra/stub/stub_asr_server.py --port 8000 --slots 8 --service-ms 400

Then point the real load generator at it:

    python bench/loadgen.py --url http://localhost:8000 --profile spike-8x --rps 5 \\
        --audio-dir golden/audio --out results/stub_spike.json

NOT a simulator of Whisper. It tells you nothing about throughput or latency on real
hardware. It tells you whether your autoscaler reacts correctly to a queue.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Engine:
    """A fixed pool of batch slots with a waiting queue — the shape of a batching server."""

    def __init__(self, slots: int, service_ms: float, jitter: float, ramp_s: float):
        self.slots = threading.Semaphore(slots)
        self.n_slots = slots
        self.service_ms = service_ms
        self.jitter = jitter
        self.started = time.time()
        self.ramp_s = ramp_s  # simulated model-load time: refuse readiness until elapsed

        self.lock = threading.Lock()
        self.waiting = 0
        self.running = 0
        self.finished = 0
        self.errors = 0
        self.total_wait_ms = 0.0

    @property
    def ready(self) -> bool:
        return (time.time() - self.started) >= self.ramp_s

    def handle(self) -> tuple[float, float]:
        """Returns (wait_ms, service_ms). Blocks like a real queued request would."""
        with self.lock:
            self.waiting += 1

        t_queued = time.perf_counter()
        self.slots.acquire()
        wait_ms = (time.perf_counter() - t_queued) * 1000

        with self.lock:
            self.waiting -= 1
            self.running += 1
            self.total_wait_ms += wait_ms

        try:
            # service time varies per request because transcript length varies — this is
            # exactly why static batching is wrong for ASR, so the stub reproduces it.
            svc = self.service_ms * random.uniform(1 - self.jitter, 1 + self.jitter)
            time.sleep(svc / 1000.0)
            return wait_ms, svc
        finally:
            with self.lock:
                self.running -= 1
                self.finished += 1
            self.slots.release()

    def snapshot(self) -> dict:
        with self.lock:
            # GPU utilization is derived from slot occupancy and SATURATES at 100 - which is
            # the whole point of the demo. Once every slot is busy this reads ~100 whether
            # the queue holds 2 requests or 2000. Queue depth keeps climbing; util cannot.
            util = min(100.0, 100.0 * self.running / self.n_slots)
            return {
                "waiting": self.waiting,
                "running": self.running,
                "finished": self.finished,
                "errors": self.errors,
                "slots": self.n_slots,
                "gpu_util": round(util, 1),
                "ready": self.ready,
            }


ENGINE: Engine  # set in main()

_LOREM = [
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    "while", "the", "committee", "reviewed", "the", "quarterly", "figures",
    "and", "adjourned", "until", "the", "following", "morning",
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # silence per-request logging; it distorts timing
        pass

    # ---------------------------------------------------------------- responses
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/metrics":
            s = ENGINE.snapshot()
            # Names VERIFIED against a live vllm-omni 0.26.0 server under load
            # (M3, 2026-08-15): the prefix is `vllm:`. Counted on the real engine —
            # 334 sample lines starting `vllm:`, ZERO starting `vllm_omni:`.
            #
            # The trap: /metrics ALSO contains `# HELP vllm_omni:num_requests_waiting`
            # and `# TYPE vllm_omni:num_requests_waiting gauge` — declarations with no
            # series behind them. So `curl /metrics | grep num_requests_waiting` finds
            # the vllm_omni name, it looks authoritative, and a KEDA query built on it
            # returns an empty result forever. The autoscaler then holds at
            # minReplicaCount through a spike with every dashboard green.
            #
            # Grep for the metric NAME and you get the wrong answer. Grep for a
            # metric LINE WITH A VALUE (`^vllm.*} [0-9]`) and you get the right one,
            # and only while traffic is flowing — an idle engine publishes neither.
            body = f"""# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model_name="stub"}} {s['waiting']}
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="stub"}} {s['running']}
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{{model_name="stub"}} {s['finished']}
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization percent (simulated; saturates at 100).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{{gpu="0"}} {s['gpu_util']}
"""
            self._send(200, body.encode(), "text/plain; version=0.0.4")
            return

        if path == "/health":
            self._json(200, {"status": "ok"})
            return

        if path == "/ready":
            # Readiness gated on simulated model-load time. Without this, Kubernetes routes
            # traffic to a pod that has not loaded weights yet, and a scale-up event shows
            # up as a burst of errors instead of added capacity - a real failure mode worth
            # being able to reproduce on a laptop.
            if ENGINE.ready:
                self._json(200, {"status": "ready"})
            else:
                self._json(503, {"status": "loading"})
            return

        if path == "/debug":
            self._json(200, ENGINE.snapshot())
            return

        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found"})
            return

        if not ENGINE.ready:
            self._json(503, {"error": "model still loading"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        remaining = length
        while remaining > 0:  # drain the upload; the bytes are irrelevant, the timing is not
            remaining -= len(self.rfile.read(min(65536, remaining)))

        wait_ms, svc_ms = ENGINE.handle()
        n = random.randint(15, 120)  # transcript length varies wildly - see build_golden.py
        text = " ".join(random.choice(_LOREM) for _ in range(n))
        self._json(200, {
            "text": text,
            "_stub": {"wait_ms": round(wait_ms, 1), "service_ms": round(svc_ms, 1)},
        })


def reporter(interval: float) -> None:
    while True:
        time.sleep(interval)
        s = ENGINE.snapshot()
        print(f"  waiting={s['waiting']:<5} running={s['running']:<4}/{s['slots']} "
              f"gpu_util={s['gpu_util']:<6} finished={s['finished']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=8,
                    help="concurrent batch slots — the pod's capacity")
    ap.add_argument("--service-ms", type=float, default=400.0,
                    help="base per-request service time once a slot is acquired")
    ap.add_argument("--jitter", type=float, default=0.4,
                    help="+/- fraction of service time, models transcript-length variance")
    ap.add_argument("--load-seconds", type=float, default=0.0,
                    help="simulated model-load time before /ready returns 200")
    ap.add_argument("--report-every", type=float, default=2.0)
    args = ap.parse_args()

    global ENGINE
    ENGINE = Engine(args.slots, args.service_ms, args.jitter, args.load_seconds)

    threading.Thread(target=reporter, args=(args.report_every,), daemon=True).start()

    theoretical = args.slots / (args.service_ms / 1000.0)
    print(f"stub ASR server on :{args.port}")
    print(f"  {args.slots} slots x {args.service_ms}ms  ->  ~{theoretical:.1f} req/s capacity")
    print("  queue depth climbs above that rate; GPU util saturates at 100 and stops informing")
    if args.load_seconds:
        print(f"  /ready returns 503 for the first {args.load_seconds}s (simulated model load)")

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
