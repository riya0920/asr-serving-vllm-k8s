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
    ap.add_argument("--dataset", default="openslr/librispeech_asr",
                    help="namespaced repo id — huggingface_hub rejects bare canonical names")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets soundfile numpy librosa")

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Streaming, so the 6GB test-clean archive is never fully downloaded — we consume ~25
    # minutes of audio and stop. Candidates are tried in order because the canonical
    # LibriSpeech repo has been renamed and restructured; pinning one path makes this script
    # break silently a year from now.
    candidates = [
        (args.dataset, args.config, args.split),
        (args.dataset, args.config, f"{args.split}.{args.config}"),
        ("openslr/librispeech_asr", "clean", "test"),
        ("librispeech_asr", "clean", "test.clean"),
    ]
    ds = None
    for repo, config, split in candidates:
        try:
            print(f"loading {repo} config={config} split={split} (streaming)...")
            ds = load_dataset(repo, config, split=split, streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  no: {type(e).__name__}: {str(e)[:120]}")
    if ds is None:
        sys.exit("could not load any LibriSpeech variant — check network and datasets version")

    SR = 16000  # Whisper's rate. Never hardcode this elsewhere — read it from the clip.
    target = int(args.seconds * SR)

    manifest = []
    buf_audio: list = []
    buf_text: list[str] = []
    buf_len = 0
    made = 0
    skipped_long = 0

    def emit() -> None:
        """Write the buffered utterances as one clip, silence-padded to exactly 30s.

        NEVER splits an utterance. An earlier version filled the buffer past 30s and then cut
        the audio at exactly 30s, which left a fragment of unreferenced speech at the end of
        every clip; Whisper faithfully transcribed it and every one of those words scored as
        an insertion. Measured baseline WER was 0.1504 with only 45 substitutions in 3631
        words — the model was ~1.2% wrong and the reference construction supplied the other
        14 points. A quality gate calibrated against that number would have been worthless.

        Padding with silence rather than trimming keeps every request exactly one 30s encoder
        window, so latency and throughput stay comparable across clips, while the reference
        describes precisely the speech present in the audio.
        """
        nonlocal made
        speech = np.concatenate(buf_audio)
        clip = np.zeros(target, dtype=np.float32)
        clip[: len(speech)] = speech
        name = f"clip_{made:03d}.wav"
        sf.write(audio_dir / name, clip, SR)
        manifest.append({
            "file": name,
            "seconds": round(len(clip) / SR, 3),
            "speech_seconds": round(len(speech) / SR, 3),
            "sampling_rate": SR,
            "reference": " ".join(buf_text),
            "utterances": len(buf_text),
        })
        made += 1
        print(f"  wrote {name}  ({len(speech)/SR:5.1f}s speech, {len(buf_text)} utterances)")

    for row in ds:
        if made >= args.clips:
            break
        wave = row["audio"]["array"]
        if row["audio"]["sampling_rate"] != SR:
            import librosa
            wave = librosa.resample(wave, orig_sr=row["audio"]["sampling_rate"], target_sr=SR)

        if len(wave) > target:
            # A single utterance longer than the encoder window cannot be represented without
            # truncating either audio or reference. Drop it rather than create a clip whose
            # reference describes speech the model never hears.
            skipped_long += 1
            continue

        if buf_len + len(wave) > target:
            emit()
            buf_audio, buf_text, buf_len = [], [], 0
            if made >= args.clips:
                break

        buf_audio.append(wave)
        buf_text.append(row["text"].strip())
        buf_len += len(wave)

    if made < args.clips:
        print(f"warning: only produced {made}/{args.clips} clips — split exhausted")
    if skipped_long:
        print(f"skipped {skipped_long} utterances longer than {args.seconds}s")

    (args.out / "manifest.json").write_text(json.dumps({
        "source": f"{args.dataset}/{args.config}/{args.split}",
        "sampling_rate": SR,
        "target_seconds": args.seconds,
        "construction": "whole utterances only, silence-padded to target; never split",
        "clips": manifest,
    }, indent=2))

    total_min = sum(c["seconds"] for c in manifest) / 60
    speech_min = sum(c["speech_seconds"] for c in manifest) / 60
    print(f"\n{made} clips, {total_min:.1f} min total "
          f"({speech_min:.1f} min speech, {100*speech_min/total_min:.0f}% speech density) "
          f"-> {args.out}")
    print("commit golden/manifest.json; audio files are large — keep them out of git "
          "(git-lfs or regenerate from this script in CI).")


if __name__ == "__main__":
    main()
