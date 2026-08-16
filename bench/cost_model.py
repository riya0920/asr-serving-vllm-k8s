#!/usr/bin/env python3
"""M9 — cost per audio hour, derived from measured throughput.

The useful property here: cost per audio hour is proportional to GPU-hours per audio-hour,
so the RATIO between two configurations is independent of what you actually pay per GPU-hour.
A ratio computed this way survives a change of cloud provider, a spot-price swing, or a
different GPU, as long as both numbers were measured on the same hardware. Both were.

    python bench/cost_model.py --baseline results/m2_baseline_a40_cuda13.json \\
        --current results/m3_turbo_latency.json --out results/m9_cost_model.json

The headroom adjustment is the honest part. Raw throughput ratio overstates the saving
because a fleet that absorbs spikes cannot run at 100% occupancy: KEDA holds steady-state
pods at ~70% of capacity so the first seconds of a spike land on warm GPUs, and
minReplicaCount keeps a floor of pods alive through idle periods. Both are real money spent
to buy latency, and both belong in the number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audio_hours_per_gpu_hour(rps: float, clip_seconds: float) -> float:
    """One GPU running for one hour transcribes this many hours of audio."""
    return rps * clip_seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-rps", type=float, required=True,
                    help="naive sequential throughput, same hardware")
    ap.add_argument("--current-rps", type=float, required=True,
                    help="optimized throughput, same hardware")
    ap.add_argument("--clip-seconds", type=float, default=30.0)
    ap.add_argument("--headroom", type=float, default=0.70,
                    help="steady-state occupancy the autoscaler holds (0.70 = 70%%)")
    ap.add_argument("--min-replicas", type=int, default=2)
    ap.add_argument("--mean-replicas", type=float, default=4.0,
                    help="average fleet size over a duty cycle, for the idle-floor term")
    ap.add_argument("--gpu-hour-usd", type=float, default=None,
                    help="optional: absolute cost, if you know your rate")
    ap.add_argument("--out", type=Path, default=Path("results/m9_cost_model.json"))
    args = ap.parse_args()

    base = audio_hours_per_gpu_hour(args.baseline_rps, args.clip_seconds)
    cur = audio_hours_per_gpu_hour(args.current_rps, args.clip_seconds)

    raw_reduction = 1 - (base / cur)

    # A spike-absorbing fleet cannot bill at full occupancy.
    effective = cur * args.headroom
    after_headroom = 1 - (base / effective)

    # The always-on floor: minReplicaCount pods stay up through idle periods. Their cost is
    # spread over the same audio volume, so it scales the effective rate down further.
    floor_penalty = args.min_replicas / args.mean_replicas
    effective_with_floor = effective * (1 - floor_penalty * 0.25)
    after_floor = 1 - (base / effective_with_floor)

    artifact = {
        "milestone": "M9",
        "method": "cost per audio hour is proportional to GPU-hours per audio-hour; the "
                  "ratio is independent of the price paid per GPU-hour",
        "clip_seconds": args.clip_seconds,
        "baseline": {
            "rps": args.baseline_rps,
            "audio_hours_per_gpu_hour": round(base, 1),
            "realtime_factor": round(base, 1),
        },
        "current": {
            "rps": args.current_rps,
            "audio_hours_per_gpu_hour": round(cur, 1),
            "realtime_factor": round(cur, 1),
        },
        "throughput_ratio": round(cur / base, 2),
        "cost_reduction": {
            "raw": round(raw_reduction, 4),
            "after_warm_headroom": round(after_headroom, 4),
            "after_headroom_and_idle_floor": round(after_floor, 4),
        },
        "assumptions": {
            "headroom_occupancy": args.headroom,
            "min_replicas": args.min_replicas,
            "mean_replicas": args.mean_replicas,
            "note": "headroom and floor are DESIGN parameters from infra/keda/scaledobject.yaml, "
                    "not measurements. The throughput numbers they scale ARE measured.",
        },
    }

    if args.gpu_hour_usd:
        artifact["absolute"] = {
            "gpu_hour_usd": args.gpu_hour_usd,
            "baseline_usd_per_audio_hour": round(args.gpu_hour_usd / base, 4),
            "current_usd_per_audio_hour": round(args.gpu_hour_usd / effective_with_floor, 4),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))

    print(f"  clip length            {args.clip_seconds}s")
    print(f"  baseline               {args.baseline_rps} req/s = {base:.0f}x realtime")
    print(f"  current                {args.current_rps} req/s = {cur:.0f}x realtime")
    print(f"  throughput ratio       {cur/base:.1f}x")
    print()
    print(f"  cost reduction, raw                        {raw_reduction:.1%}")
    print(f"  ...after {args.headroom:.0%} warm headroom                {after_headroom:.1%}")
    print(f"  ...after headroom + idle floor             {after_floor:.1%}")
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
