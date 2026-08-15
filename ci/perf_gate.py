#!/usr/bin/env python3
"""Fail the build when serving got slower.

The WER gate catches "correct but broken". This catches "correct but slow" — a change that
transcribes perfectly and drops throughput 40% is also a failed deployment, and no other
check in the pipeline would say a word about it.

Two independent assertions, because they fail for different reasons:

  --max-p99-ms       an absolute SLO. Non-negotiable regardless of history.
  --max-regression   a relative check against the last recorded good run. Catches slow
                     drift that never individually breaches the SLO but eats the headroom
                     the autoscaler depends on.

    python ci/perf_gate.py --current results/ci_load.json \\
        --baseline results/m3_vllm_batching.json --max-p99-ms 620 --max-regression 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def worst_phase(artifact: dict) -> dict:
    """Take the worst phase, not the average across phases.

    A spike-profile run has a quiet 'pre' phase and an overloaded 'spike' phase. Averaging
    them produces a number that describes no moment the service ever experienced, and
    flatters exactly the scenario the SLO exists for.
    """
    phases = artifact.get("phases") or []
    if not phases:
        raise SystemExit("artifact has no phases")
    return max(phases, key=lambda p: p.get("p99_ms", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--max-p99-ms", type=float, default=620.0)
    ap.add_argument("--max-regression", type=float, default=0.10,
                    help="fractional throughput drop allowed vs baseline (0.10 = 10%%)")
    args = ap.parse_args()

    cur = json.loads(args.current.read_text())
    phase = worst_phase(cur)

    print(f"current run: {args.current}")
    print(f"  worst phase   {phase['phase']}")
    print(f"  requests      {phase['requests']} ({phase['errors']} errors)")
    print(f"  throughput    {phase['achieved_rps']} req/s")
    print(f"  p50 / p99     {phase['p50_ms']} / {phase['p99_ms']} ms")

    failed = []

    # A run where the load generator itself lagged measures the generator, not the server.
    # Passing such a run would be worse than failing it: it looks like evidence.
    if not phase.get("generator_healthy", True):
        failed.append(
            f"load generator lagged {phase.get('max_queue_delay_ms')}ms — run is invalid, "
            f"not slow. Re-run from a machine that is not under test."
        )

    if phase["errors"] > 0:
        rate = phase["errors"] / max(1, phase["requests"])
        if rate > 0.01:
            failed.append(f"error rate {rate:.2%} exceeds 1%")

    if phase["p99_ms"] > args.max_p99_ms:
        failed.append(f"p99 {phase['p99_ms']}ms exceeds SLO {args.max_p99_ms}ms")

    if args.baseline and args.baseline.exists():
        base = worst_phase(json.loads(args.baseline.read_text()))
        base_rps = base.get("achieved_rps", 0)
        if base_rps > 0:
            drop = (base_rps - phase["achieved_rps"]) / base_rps
            print(f"\nbaseline: {args.baseline}")
            print(f"  throughput    {base_rps} req/s")
            print(f"  change        {-drop:+.1%}")
            if drop > args.max_regression:
                failed.append(
                    f"throughput dropped {drop:.1%} vs baseline "
                    f"(limit {args.max_regression:.0%})"
                )
    elif args.baseline:
        # Not fatal: the very first run has nothing to compare against. But say so loudly,
        # because "no baseline" and "no regression" print almost the same green checkmark.
        print(f"\nWARNING: baseline {args.baseline} not found — "
              f"absolute SLO checked, regression NOT checked")

    print()
    if failed:
        for f in failed:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
