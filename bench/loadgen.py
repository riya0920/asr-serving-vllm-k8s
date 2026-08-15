#!/usr/bin/env python3
"""Open-loop load generator for the ASR serving stack.

Drives an OpenAI-compatible /v1/audio/transcriptions endpoint and records latency
percentiles per phase, so the same tool produces the steady-state number (M3) and
the 8x spike number (M6).

Two design decisions that make the numbers honest:

1. OPEN LOOP. Requests are launched on a fixed arrival schedule regardless of whether
   earlier ones have returned. A closed-loop generator (N workers each sending the next
   request only after the previous returns) throttles itself exactly when the server
   slows down, which is the classic coordinated-omission bug: the server melts and the
   generator politely stops asking, so p99 looks fine. If you are trying to prove an
   autoscaler absorbs a spike, closed loop will lie to you in the flattering direction.

2. LATENCY IS MEASURED FROM SCHEDULED SEND TIME, not from actual send time. If the
   generator itself falls behind, that queueing delay counts against the measurement
   rather than being silently discarded. `queue_delay_ms` in the output reports how much
   of the total came from generator lag — if it is not near zero, the generator is the
   bottleneck and the run is invalid.

Usage:
    python bench/loadgen.py --url http://localhost:8000 --audio-dir golden/audio \\
        --profile spike-8x --out results/m6_spike_8x.json

Profiles:
    steady   --rps R --duration S
    sweep    --concurrency-list 1,2,4,8,...   (closed loop on purpose; for M3 batch sweep)
    spike-8x --rps R  (base R for 60s, step to 8R within 2s, hold 120s, back to R for 120s)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")


# --------------------------------------------------------------------------------------
# results


@dataclass
class Sample:
    phase: str
    scheduled_at: float      # when it SHOULD have been sent (relative to run start)
    sent_at: float
    finished_at: float
    status: int
    ok: bool
    chars: int = 0
    error: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.finished_at - self.scheduled_at) * 1000.0

    @property
    def queue_delay_ms(self) -> float:
        return (self.sent_at - self.scheduled_at) * 1000.0


@dataclass
class PhaseStats:
    phase: str
    requests: int
    errors: int
    duration_s: float
    achieved_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_queue_delay_ms: float
    max_queue_delay_ms: float
    generator_healthy: bool = field(default=True)


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value at or above which q of the samples fall.

    math.ceil, not round() — Python's round() is banker's rounding, so round(5.5) is 6 and
    p50 of 10 samples would return the 6th value instead of the 5th. Nearest rank also means
    p99 of 50 samples is the max, which is honest: with 50 samples you have not measured p99
    and the number should not pretend otherwise.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def summarize(samples: list[Sample]) -> list[PhaseStats]:
    out: list[PhaseStats] = []
    phases: dict[str, list[Sample]] = {}
    for s in samples:
        phases.setdefault(s.phase, []).append(s)

    for phase, group in phases.items():
        good = [s for s in group if s.ok]
        lat = [s.latency_ms for s in good]
        qd = [s.queue_delay_ms for s in group]
        span = max(s.finished_at for s in group) - min(s.scheduled_at for s in group)
        max_qd = max(qd) if qd else 0.0
        out.append(
            PhaseStats(
                phase=phase,
                requests=len(group),
                errors=len(group) - len(good),
                duration_s=round(span, 2),
                achieved_rps=round(len(group) / span, 2) if span > 0 else 0.0,
                p50_ms=round(pct(lat, 0.50), 1),
                p95_ms=round(pct(lat, 0.95), 1),
                p99_ms=round(pct(lat, 0.99), 1),
                max_ms=round(max(lat), 1) if lat else float("nan"),
                mean_queue_delay_ms=round(statistics.fmean(qd), 1) if qd else 0.0,
                max_queue_delay_ms=round(max_qd, 1),
                # if the generator itself added >50ms, the run does not measure the server
                generator_healthy=max_qd < 50.0,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# request


async def transcribe(
    client: httpx.AsyncClient,
    url: str,
    audio: tuple[str, bytes],
    model: str,
    phase: str,
    scheduled_at: float,
    t0: float,
    results: list[Sample],
) -> None:
    name, blob = audio
    sent = time.perf_counter() - t0
    status, ok, chars, err = 0, False, 0, ""
    try:
        r = await client.post(
            f"{url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": (name, blob, "audio/wav")},
            data={"model": model, "response_format": "json"},
        )
        status = r.status_code
        ok = r.status_code == 200
        if ok:
            chars = len((r.json() or {}).get("text", ""))
        else:
            err = r.text[:200]
    except Exception as e:  # noqa: BLE001 - a failed request is data, not a crash
        err = f"{type(e).__name__}: {e}"[:200]
    finally:
        results.append(
            Sample(
                phase=phase,
                scheduled_at=scheduled_at,
                sent_at=sent,
                finished_at=time.perf_counter() - t0,
                status=status,
                ok=ok,
                chars=chars,
                error=err,
            )
        )


# --------------------------------------------------------------------------------------
# profiles


async def run_open_loop(
    schedule: list[tuple[float, str]],  # (scheduled_offset_s, phase)
    url: str,
    model: str,
    clips: list[tuple[str, bytes]],
    timeout: float,
) -> list[Sample]:
    """Fire requests at their scheduled times, no matter what the server is doing."""
    results: list[Sample] = []
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        tasks: list[asyncio.Task] = []
        for offset, phase in schedule:
            now = time.perf_counter() - t0
            if offset > now:
                await asyncio.sleep(offset - now)
            clip = random.choice(clips)
            tasks.append(
                asyncio.create_task(
                    transcribe(client, url, clip, model, phase, offset, t0, results)
                )
            )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return results


def poisson_schedule(rps: float, duration: float, start: float, phase: str, rng: random.Random):
    """Poisson arrivals — real traffic is not evenly spaced, and even spacing understates p99."""
    t, out = start, []
    end = start + duration
    while t < end:
        t += rng.expovariate(rps)
        if t < end:
            out.append((t, phase))
    return out


def build_schedule(profile: str, rps: float, duration: float, seed: int):
    rng = random.Random(seed)
    if profile == "steady":
        return poisson_schedule(rps, duration, 0.0, "steady", rng)

    if profile == "spike-8x":
        # Step function, NOT a ramp. A ramp gives the autoscaler time it will not get
        # at the top of the hour when every meeting starts at once.
        sched = []
        sched += poisson_schedule(rps, 60.0, 0.0, "pre", rng)          # settle
        sched += poisson_schedule(rps * 8, 120.0, 62.0, "spike", rng)  # 8x, 2s transition
        sched += poisson_schedule(rps, 120.0, 182.0, "recover", rng)   # scale back down
        return sorted(sched)

    raise SystemExit(f"unknown profile: {profile}")


async def run_sweep(url, model, clips, concurrency_list, per_level, timeout):
    """Closed loop on purpose: finds the concurrency at which p99 crosses the SLO (M3)."""
    out = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for c in concurrency_list:
            results: list[Sample] = []
            t0 = time.perf_counter()

            async def worker():
                while len(results) < per_level:
                    at = time.perf_counter() - t0
                    await transcribe(
                        client, url, random.choice(clips), model, f"c={c}", at, t0, results
                    )

            await asyncio.gather(*[worker() for _ in range(c)])
            stats = summarize(results)[0]
            stats.generator_healthy = True  # closed loop: queue delay is meaningless here
            print(
                f"  concurrency={c:<4} rps={stats.achieved_rps:<8} "
                f"p50={stats.p50_ms:<8} p99={stats.p99_ms:<9} errors={stats.errors}"
            )
            out.append(stats)
    return out


# --------------------------------------------------------------------------------------


def load_clips(audio_dir: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in audio_dir.glob("*.wav"))
    if not files:
        raise SystemExit(f"no .wav files in {audio_dir} — build the golden set first (M2)")
    clips = [(p.name, p.read_bytes()) for p in files]
    total_mb = sum(len(b) for _, b in clips) / 1e6
    print(f"loaded {len(clips)} clips ({total_mb:.1f} MB) into memory")
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--audio-dir", type=Path, default=Path("golden/audio"))
    ap.add_argument("--profile", default="steady", choices=["steady", "spike-8x", "sweep"])
    ap.add_argument("--rps", type=float, default=5.0, help="base arrival rate (spike goes to 8x this)")
    ap.add_argument("--duration", type=float, default=120.0, help="steady profile only")
    ap.add_argument("--concurrency-list", default="1,2,4,8,16,32,64", help="sweep profile only")
    ap.add_argument("--per-level", type=int, default=200, help="requests per sweep level")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="", help="free text recorded in the artifact (hardware, commit, etc.)")
    args = ap.parse_args()

    clips = load_clips(args.audio_dir)
    started = time.time()

    if args.profile == "sweep":
        levels = [int(x) for x in args.concurrency_list.split(",")]
        print(f"sweeping concurrency {levels}, {args.per_level} requests each")
        stats = asyncio.run(run_sweep(args.url, args.model, clips, levels, args.per_level, args.timeout))
        samples_n = sum(s.requests for s in stats)
    else:
        schedule = build_schedule(args.profile, args.rps, args.duration, args.seed)
        print(f"profile={args.profile}  scheduled {len(schedule)} requests over "
              f"{schedule[-1][0]:.0f}s  (open loop, Poisson arrivals)")
        samples = asyncio.run(run_open_loop(schedule, args.url, args.model, clips, args.timeout))
        stats = summarize(samples)
        samples_n = len(samples)

    artifact = {
        "profile": args.profile,
        "url": args.url,
        "model": args.model,
        "base_rps": args.rps,
        "clips": len(clips),
        "requests": samples_n,
        "started_unix": started,
        "note": args.note,
        "phases": [asdict(s) for s in stats],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'phase':<10} {'req':>6} {'err':>5} {'rps':>8} {'p50':>9} {'p95':>9} {'p99':>9}")
    for s in stats:
        print(f"{s.phase:<10} {s.requests:>6} {s.errors:>5} {s.achieved_rps:>8} "
              f"{s.p50_ms:>9} {s.p95_ms:>9} {s.p99_ms:>9}")
        if not s.generator_healthy:
            print(f"  !! phase '{s.phase}': generator lagged {s.max_queue_delay_ms:.0f}ms — "
                  f"this run measures the load generator, not the server. Discard it.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
