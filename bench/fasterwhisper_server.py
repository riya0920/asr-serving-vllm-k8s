#!/usr/bin/env python3
"""faster-whisper (CTranslate2) behind the same HTTP contract as vLLM.

WHY THIS EXISTS
---------------
Every measurement so far says vLLM is the constraint for Whisper, not the hardware:

  A40   13.07 req/s, GPU saturated at 100% / 300 W
  H100   5.93 req/s, GPU at 23-30% / 126 W, engine burning 23 CPU cores
  after moving mel to the GPU: 16/192 cores busy, GPU 50% at 130 W

High utilization at low power with neither resource maxed is the signature of a serialized
per-request path, not a throughput ceiling. CTranslate2 is a different implementation
entirely — purpose-built for Whisper, its own batching, its own kernels — so it does not
share that path. If it is several times faster on identical hardware with identical audio,
then the throughput target was always reachable and vLLM was the wrong engine for this model.

The comparison is only worth anything if NOTHING else changes. This server therefore speaks
the same OpenAI-compatible endpoint, exposes the same Prometheus metric names, and is driven
by the same bench/loadgen.py and graded by the same ci/wer_gate.py. Same clips, same
harness, same gate — one variable.

    python bench/fasterwhisper_server.py --port 8000 --model large-v3-turbo --batch-size 32

Stdlib HTTP on purpose: adding FastAPI/uvicorn here would introduce a second variable
(a different web stack) into a benchmark whose entire point is isolating the engine.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = None
PIPELINE = None
STATE = {"waiting": 0, "running": 0, "finished": 0, "failed": 0}
LOCK = threading.Lock()
SLOTS: threading.Semaphore
N_SLOTS = 32


def transcribe(wav_bytes: bytes) -> str:
    """Run one clip. Blocks on a slot so queue depth is observable, exactly like vLLM."""
    with LOCK:
        STATE["waiting"] += 1
    SLOTS.acquire()
    with LOCK:
        STATE["waiting"] -= 1
        STATE["running"] += 1
    try:
        buf = io.BytesIO(wav_bytes)
        # BatchedInferencePipeline is where CTranslate2's throughput advantage lives; without
        # it this is a sequential comparison and the benchmark answers the wrong question.
        engine = PIPELINE if PIPELINE is not None else MODEL
        if PIPELINE is not None:
            segments, _ = engine.transcribe(buf, batch_size=N_SLOTS, language="en")
        else:
            segments, _ = engine.transcribe(buf, language="en")
        return " ".join(s.text for s in segments).strip()
    finally:
        with LOCK:
            STATE["running"] -= 1
            STATE["finished"] += 1
        SLOTS.release()


class BenchServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a listen backlog that survives a benchmark.

    socketserver defaults request_queue_size to 5. Drive 128 concurrent clients at it and the
    kernel refuses connections once the accept queue overflows — which surfaces as client-side
    errors that look exactly like the engine failing, while the server log stays clean because
    those requests never reached it. Measured here as 126 errors in 324 requests before the
    fix, with zero server-side exceptions.
    """

    request_queue_size = 256
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # per-request logging distorts timing
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/health", "/ready"):
            self._json(200, {"status": "ok"})
            return
        if path == "/metrics":
            with LOCK:
                s = dict(STATE)
            util = min(100.0, 100.0 * s["running"] / max(1, N_SLOTS))
            # Same metric names as vLLM so the KEDA ScaledObject and Prometheus rules in
            # infra/ apply to this engine with zero changes.
            body = (
                "# TYPE vllm:num_requests_waiting gauge\n"
                f'vllm:num_requests_waiting{{model_name="faster-whisper"}} {s["waiting"]}\n'
                "# TYPE vllm:num_requests_running gauge\n"
                f'vllm:num_requests_running{{model_name="faster-whisper"}} {s["running"]}\n'
                "# TYPE vllm:request_success_total counter\n"
                f'vllm:request_success_total{{model_name="faster-whisper"}} {s["finished"]}\n'
                "# TYPE vllm:request_failure_total counter\n"
                f'vllm:request_failure_total{{model_name="faster-whisper"}} {s["failed"]}\n'
                "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
                f'DCGM_FI_DEV_GPU_UTIL{{gpu="0"}} {util:.1f}\n'
            )
            self._send(200, body.encode(), "text/plain; version=0.0.4")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = b""
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            raw += chunk
            remaining -= len(chunk)

        # Minimal multipart extraction: find the WAV payload between boundaries. Using a
        # full multipart library would drag in another dependency for no benchmark value.
        body = raw
        idx = raw.find(b"RIFF")
        if idx != -1:
            end = raw.rfind(b"\r\n--")
            body = raw[idx:end] if end > idx else raw[idx:]

        try:
            text = transcribe(body)
            self._json(200, {"text": text})
        except Exception as e:  # noqa: BLE001 - a failed request is data, not a crash
            with LOCK:
                STATE["failed"] += 1
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global MODEL, PIPELINE, SLOTS, N_SLOTS
    N_SLOTS = args.batch_size
    SLOTS = threading.Semaphore(N_SLOTS)

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    MODEL = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    try:
        from faster_whisper import BatchedInferencePipeline

        PIPELINE = BatchedInferencePipeline(model=MODEL)
        mode = "batched"
    except ImportError:
        PIPELINE = None
        mode = "sequential (BatchedInferencePipeline unavailable — comparison is unfair, say so)"
    print(f"loaded {args.model} in {time.perf_counter()-t0:.1f}s [{mode}], {N_SLOTS} slots")
    print(f"faster-whisper server on :{args.port}")

    BenchServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
