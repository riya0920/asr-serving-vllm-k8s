#!/usr/bin/env python3
"""M2 — the naive baseline. Whisper-Large-v3, HF transformers, one request at a time.

This is the denominator. Every speedup claim in this project is measured against the JSON
this script writes, on the same GPU and the same golden set. If the hardware changes, the
baseline is re-run; a speedup measured against a baseline from different hardware is not a
speedup, it is a hardware comparison wearing a costume.

    python bench/baseline_sequential.py --golden golden --out results/m2_baseline_sequential.json

Records both throughput AND word error rate. The WER here becomes the reference value the
CI gate (M7) compares against — a change that makes serving faster and transcripts worse
must fail, and it cannot fail if nobody wrote down what "before" was.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wer import corpus_wer


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": "cpu", "note": "no CUDA — baseline will be meaningless for serving"}
        i = torch.cuda.get_device_properties(0)
        return {
            "device": i.name,
            "vram_gb": round(i.total_memory / 1e9, 1),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "count": torch.cuda.device_count(),
        }
    except ImportError:
        return {"device": "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--golden", type=Path, default=Path("golden"))
    ap.add_argument("--out", type=Path, default=Path("results/m2_baseline_sequential.json"))
    ap.add_argument("--warmup", type=int, default=3, help="discarded runs before timing starts")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    # On H100 + torch 2.11, cuDNN's scaled-dot-product-attention backend has no execution
    # plan for Whisper's attention shapes and dies with "cudnn_frontend Error: No valid
    # execution plans built" the moment the encoder runs. Disabling that one backend leaves
    # flash and mem-efficient SDPA available, so this costs nothing measurable and keeps the
    # baseline running on the same code path across GPUs. Did not reproduce on A40.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    manifest = json.loads((args.golden / "manifest.json").read_text())
    clips = manifest["clips"]
    if not clips:
        raise SystemExit("golden set is empty — run bench/build_golden.py first")

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype})...")
    load_t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to("cuda").eval()
    load_s = time.perf_counter() - load_t
    print(f"loaded in {load_s:.1f}s  (this is the number that dominates pod cold start — M5)")

    audio = []
    for c in clips:
        wave, sr = sf.read(args.golden / "audio" / c["file"], dtype="float32")
        assert sr == c["sampling_rate"], f"{c['file']}: manifest says {c['sampling_rate']}, file says {sr}"
        audio.append(wave)

    # warmup: first calls include CUDA context setup and kernel autotuning. Timing them
    # makes the baseline look worse than it is, which would flatter every later number.
    print(f"warming up ({args.warmup} runs, discarded)...")
    for i in range(args.warmup):
        feats = processor(audio[i % len(audio)], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            model.generate(feats.input_features.to("cuda", dtype), max_new_tokens=440)
    torch.cuda.synchronize()

    print(f"timing {len(clips)} clips, strictly sequential...")
    latencies, hyps, tokens = [], [], []
    wall_start = time.perf_counter()

    # strict=True: if the manifest and the loaded audio ever disagree in length, fail loudly.
    # A silent truncation here would time fewer clips than the manifest claims and report a
    # throughput number computed over the wrong denominator.
    for c, wave in zip(clips, audio, strict=True):
        t = time.perf_counter()
        feats = processor(wave, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to("cuda", dtype), max_new_tokens=440)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t) * 1000)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        hyps.append(text)
        tokens.append(int(ids.shape[-1]))
        print(f"  {c['file']}  {latencies[-1]:7.0f} ms  {tokens[-1]:4d} tok")

    wall = time.perf_counter() - wall_start
    audio_s = sum(c["seconds"] for c in clips)
    quality = corpus_wer([(c["reference"], h) for c, h in zip(clips, hyps, strict=True)])

    artifact = {
        "milestone": "M2",
        "kind": "sequential_baseline",
        "model": args.model,
        "dtype": args.dtype,
        "hardware": gpu_info(),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "model_load_seconds": round(load_s, 2),
        "clips": len(clips),
        "audio_seconds": round(audio_s, 1),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(clips) / wall, 3),
        "realtime_factor": round(audio_s / wall, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1),
            "p50": round(statistics.median(latencies), 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "decoded_tokens": {
            "mean": round(statistics.fmean(tokens), 1),
            "min": min(tokens),
            "max": max(tokens),
        },
        "quality": quality,
        "note": args.note,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'':-<58}")
    print(f"  throughput      {artifact['throughput_rps']} req/s   (1 GPU, sequential)")
    print(f"  realtime factor {artifact['realtime_factor']}x")
    print(f"  p50 latency     {artifact['latency_ms']['p50']} ms")
    print(f"  corpus WER      {quality['wer']:.4f}   <- CI gate baseline")
    print(f"  tokens/clip     {artifact['decoded_tokens']['mean']} "
          f"(range {artifact['decoded_tokens']['min']}-{artifact['decoded_tokens']['max']})")
    print(f"{'':-<58}")
    print("token spread is why static batching is wrong for ASR: in a static batch every")
    print("short transcript waits for the longest one in the batch.")
    print(f"\nwrote {args.out}  -> update the README scoreboard from this file")


if __name__ == "__main__":
    main()
