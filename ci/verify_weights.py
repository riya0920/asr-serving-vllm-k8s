#!/usr/bin/env python3
"""Verify model weights are exactly what we think they are, before anything loads them.

Runs ahead of the WER gate because it answers a different question far more cheaply. The WER
gate asks "does this model transcribe well?" — a 40-minute GPU job. This asks "is this the
model we intended to ship?" — seconds, and it catches the cases where the answer is no for
boring reasons: a truncated download, a floating HF revision that moved under us, a cached
layer from a different model, a partially-written file from an interrupted pull.

Any of those can still load. Some of them still transcribe, just worse. Content-addressing
the artifact removes the whole class.

    python ci/verify_weights.py --manifest model-manifest.json

Regenerate the manifest deliberately, never automatically:

    python ci/verify_weights.py --manifest model-manifest.json --update
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_dir(model_id: str, revision: str, cache: Path) -> Path:
    """Locate a snapshot in the HF cache without importing transformers."""
    repo = cache / f"models--{model_id.replace('/', '--')}"
    snap = repo / "snapshots" / revision
    if snap.is_dir():
        return snap
    snaps = repo / "snapshots"
    if snaps.is_dir():
        found = sorted(p for p in snaps.iterdir() if p.is_dir())
        if len(found) == 1:
            return found[0]
        raise SystemExit(
            f"revision '{revision}' not in cache; found {[p.name for p in found]}. "
            f"Pin the exact revision — 'main' moves, and a model that silently changed is "
            f"the hardest kind of regression to diagnose."
        )
    raise SystemExit(f"model {model_id} not found under {cache}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("model-manifest.json"))
    ap.add_argument("--cache", type=Path,
                    default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    ap.add_argument("--update", action="store_true", help="rewrite the manifest from what is on disk")
    args = ap.parse_args()

    if not args.manifest.exists() and not args.update:
        print(f"FAIL: {args.manifest} missing. Generate it with --update on a machine where "
              f"the model is known good, then commit it.", file=sys.stderr)
        return 2

    spec = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    model_id = spec.get("model_id", "openai/whisper-large-v3")
    revision = spec.get("revision", "main")

    if revision in ("main", "master", "latest"):
        # A floating revision means "whatever was published most recently", which makes the
        # build non-reproducible and the WER baseline meaningless.
        print(f"FAIL: revision is '{revision}' — pin an immutable commit SHA", file=sys.stderr)
        return 2

    snapshot = resolve_dir(model_id, revision, args.cache)
    print(f"model     {model_id}")
    print(f"revision  {revision}")
    print(f"snapshot  {snapshot}")

    if args.update:
        files = {}
        for p in sorted(snapshot.rglob("*")):
            if p.is_file() and p.suffix in {".safetensors", ".bin", ".json", ".txt", ".model"}:
                files[str(p.relative_to(snapshot)).replace("\\", "/")] = {
                    "sha256": sha256(p), "bytes": p.stat().st_size,
                }
                print(f"  hashed {p.name}")
        args.manifest.write_text(json.dumps(
            {"model_id": model_id, "revision": revision, "files": files}, indent=2))
        print(f"\nwrote {args.manifest} with {len(files)} files — review and commit it")
        return 0

    expected = spec.get("files", {})
    if not expected:
        print("FAIL: manifest lists no files", file=sys.stderr)
        return 2

    failures = []
    for name, meta in expected.items():
        path = snapshot / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        size = path.stat().st_size
        if size != meta["bytes"]:
            # Almost always a truncated or resumed download. Check size before hashing so
            # the common failure reports in milliseconds instead of after hashing 3GB.
            failures.append(f"{name}: {size} bytes, expected {meta['bytes']} (truncated?)")
            continue
        digest = sha256(path)
        if digest != meta["sha256"]:
            failures.append(f"{name}: sha256 {digest[:16]}… != {meta['sha256'][:16]}…")
        else:
            print(f"  ok  {name}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"all {len(expected)} files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
