#!/usr/bin/env python3
"""The gate that stands between a quietly-broken model and production.

Runs the golden set through a live server and fails the build if word error rate has
regressed past tolerance against the recorded baseline.

This exists because of a failure mode that no other check catches. A wrong mel-spectrogram
config, a mismatched processor version, the wrong language token, a truncated weight
download that still passes a shape check — none of these crash. The container starts, the
health probe goes green, latency looks fine, requests return 200 with plausible-looking
English in them. The only symptom is that the transcripts are worse, and without this gate
the first person to notice is a user.

    python ci/wer_gate.py --url http://localhost:8000 \\
        --baseline results/m2_baseline_sequential.json --tolerance 0.02

Exit 0 = pass, 1 = regression, 2 = could not measure (which is also a failure — a gate that
silently passes when it cannot run is worse than no gate, because it is trusted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from wer import corpus_wer, wer as clip_wer  # noqa: E402

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")


def transcribe_all(url: str, golden: Path, model: str, timeout: float) -> list[tuple[str, str, str]]:
    manifest = json.loads((golden / "manifest.json").read_text())
    out = []
    with httpx.Client(timeout=timeout) as client:
        for i, clip in enumerate(manifest["clips"], 1):
            path = golden / "audio" / clip["file"]
            r = client.post(
                f"{url.rstrip('/')}/v1/audio/transcriptions",
                files={"file": (clip["file"], path.read_bytes(), "audio/wav")},
                data={"model": model, "response_format": "json", "language": "en"},
            )
            r.raise_for_status()
            out.append((clip["file"], clip["reference"], r.json()["text"]))
            print(f"  [{i}/{len(manifest['clips'])}] {clip['file']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--golden", type=Path, default=Path("golden"))
    ap.add_argument("--baseline", type=Path, default=Path("results/m2_baseline_sequential.json"))
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="absolute WER regression allowed vs baseline (0.02 = 2 points)")
    ap.add_argument("--worst-clip-max", type=float, default=0.60,
                    help="fail if any single clip exceeds this WER, even if corpus WER passes")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", type=Path, default=Path("results/ci_wer_gate.json"))
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"FAIL: no baseline at {args.baseline}. Run M2 first — a gate with nothing to "
              f"compare against cannot fail, and would give false confidence.", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text())
    base_wer = baseline.get("quality", {}).get("wer")
    if base_wer is None:
        print(f"FAIL: {args.baseline} has no quality.wer field", file=sys.stderr)
        return 2

    print(f"baseline WER {base_wer:.4f} (from {args.baseline}, "
          f"{baseline.get('hardware', {}).get('device', 'unknown hw')})")
    print(f"transcribing golden set against {args.url}")

    try:
        results = transcribe_all(args.url, args.golden, args.model, args.timeout)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not transcribe: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if not results:
        print("FAIL: golden set is empty", file=sys.stderr)
        return 2

    quality = corpus_wer([(ref, hyp) for _, ref, hyp in results])
    delta = quality["wer"] - base_wer

    per_clip = sorted(
        ((f, clip_wer(ref, hyp)["wer"]) for f, ref, hyp in results),
        key=lambda x: -x[1],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "url": args.url,
        "model": args.model,
        "baseline_wer": base_wer,
        "measured_wer": quality["wer"],
        "delta": delta,
        "tolerance": args.tolerance,
        "corpus": quality,
        "worst_clips": per_clip[:5],
    }, indent=2))

    print(f"\n  baseline   {base_wer:.4f}")
    print(f"  measured   {quality['wer']:.4f}")
    print(f"  delta      {delta:+.4f}  (tolerance +{args.tolerance:.4f})")
    print(f"  worst clip {per_clip[0][1]:.4f}  ({per_clip[0][0]})")
    print(f"  errors     {quality['substitutions']}S {quality['deletions']}D "
          f"{quality['insertions']}I over {quality['ref_words']} words")

    failed = False
    if delta > args.tolerance:
        print(f"\nFAIL: WER regressed {delta:+.4f}, tolerance is +{args.tolerance:.4f}")
        print("  top offenders:")
        for f, w in per_clip[:5]:
            print(f"    {w:.4f}  {f}")
        failed = True

    if per_clip[0][1] > args.worst_clip_max:
        # Corpus WER can stay green while one clip collapses completely — e.g. language
        # detection picking the wrong language on a single file. Averages hide cliffs.
        print(f"\nFAIL: {per_clip[0][0]} has WER {per_clip[0][1]:.4f} "
              f"(max {args.worst_clip_max}) — a single clip failing this hard usually means "
              f"a per-request bug, not a model regression")
        failed = True

    if failed:
        return 1

    if delta < -args.tolerance:
        print(f"\nPASS — and WER IMPROVED by {-delta:.4f}. Re-baseline (rerun M2 and commit "
              f"the new artifact) so future regressions are measured against this, not the "
              f"old worse number.")
    else:
        print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
