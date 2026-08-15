#!/usr/bin/env python3
"""Build the golden set: ~50 clips of 30s speech with reference transcripts.

This set does double duty and that is intentional:
  - the benchmark corpus (M2, M3, M6) — so throughput numbers are always over the same audio
  - the CI WER gate corpus (M7) — so a model regression is caught against known references

Source is LibriSpeech test-clean: real read speech, public references, redistributable.
Individual utterances are 2-15s, so they are concatenated into 30s clips (the resume claim
is about 30s clips, and Whisper's encoder window is 30s — anything shorter is padded to 30s
anyway and would understate the decode work per request).

    python bench/build_golden.py --clips 50 --out golden

Why not synthetic audio or silence: decode length drives latency almost entirely, and silence
decodes to nearly nothing. Benchmarking on silence measures the encoder and reports numbers
that collapse the moment real speech arrives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("golden"))
    ap.add_argument("--split", default="test.clean")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets soundfile numpy librosa")

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading librispeech_asr {args.split} (streaming)...")
    ds = load_dataset("librispeech_asr", split=args.split, streaming=True)

    SR = 16000  # Whisper's rate. Never hardcode this elsewhere — read it from the clip.
    target = int(args.seconds * SR)

    manifest = []
    buf_audio: list = []
    buf_text: list[str] = []
    buf_len = 0
    made = 0

    for row in ds:
        if made >= args.clips:
            break
        wave = row["audio"]["array"]
        if row["audio"]["sampling_rate"] != SR:
            import librosa
            wave = librosa.resample(wave, orig_sr=row["audio"]["sampling_rate"], target_sr=SR)

        buf_audio.append(wave)
        buf_text.append(row["text"].strip())
        buf_len += len(wave)

        if buf_len >= target:
            clip = np.concatenate(buf_audio)[:target]
            name = f"clip_{made:03d}.wav"
            sf.write(audio_dir / name, clip, SR)
            manifest.append({
                "file": name,
                "seconds": round(len(clip) / SR, 3),
                "sampling_rate": SR,
                # reference covers the utterances that fit — the tail utterance is truncated
                # by the 30s cut, so its words are dropped from the reference too. Keeping a
                # reference for audio that is not in the clip would inflate deletions forever.
                "reference": " ".join(buf_text[:-1]) if len(buf_text) > 1 else buf_text[0],
                "utterances": len(buf_text) - 1 if len(buf_text) > 1 else 1,
            })
            made += 1
            buf_audio, buf_text, buf_len = [], [], 0
            print(f"  wrote {name}")

    if made < args.clips:
        print(f"warning: only produced {made}/{args.clips} clips — split exhausted")

    (args.out / "manifest.json").write_text(json.dumps({
        "source": f"librispeech_asr/{args.split}",
        "sampling_rate": SR,
        "target_seconds": args.seconds,
        "clips": manifest,
    }, indent=2))

    total_min = sum(c["seconds"] for c in manifest) / 60
    print(f"\n{made} clips, {total_min:.1f} minutes of audio -> {args.out}")
    print("commit golden/manifest.json; audio files are large — keep them out of git "
          "(git-lfs or regenerate from this script in CI).")


if __name__ == "__main__":
    main()
