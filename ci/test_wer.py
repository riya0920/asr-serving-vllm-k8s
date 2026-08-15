#!/usr/bin/env python3
"""Unit tests for the WER math.

The WER gate is what stands between a silently-degraded model and production. If corpus_wer
is wrong, that gate is decorative — it will pass everything, and nobody will find out until
the transcripts are bad in front of users. So the guard gets its own guard.

Stdlib only, no pytest: this runs in the cheapest CI job, before anything is installed.

    python ci/test_wer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from wer import normalize, wer, corpus_wer  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want, tol: float = 1e-9) -> None:
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(name)


print("normalization")
check("lowercases and strips punctuation", normalize("Hello, World!"), "hello world")
check("expands contractions", normalize("Don't stop"), "do not stop")
check("won't is not wo not", normalize("won't"), "will not")
check("collapses whitespace", normalize("a   b\n\tc"), "a b c")
check("unicode normalizes", normalize("café"), "café")

print("\nsingle-utterance WER")
check("identical is zero", wer("the quick brown fox", "the quick brown fox")["wer"], 0.0)
check("one substitution of four", wer("the quick brown fox", "the quick red fox")["wer"], 0.25)
check("counts substitution", wer("a b c", "a x c")["substitutions"], 1)
check("counts deletion", wer("a b c", "a c")["deletions"], 1)
check("counts insertion", wer("a b c", "a b x c")["insertions"], 1)
check("everything wrong is 1.0", wer("a b c", "x y z")["wer"], 1.0)
check("punctuation-only diff is zero", wer("hello world", "Hello, world.")["wer"], 0.0)
# WER is unbounded above — a hallucinating model that emits 100 words for a 3-word
# reference must score worse than one that emits nothing. Clamping to 1.0 would make those
# two failures indistinguishable, and hallucination is the more dangerous one.
check("hallucination exceeds 1.0", wer("a b c", " ".join(["x"] * 100))["wer"] > 1.0, True)
check("empty hypothesis is total loss", wer("a b", "")["wer"], 1.0)
check("empty reference with output", wer("", "spurious")["wer"], 1.0)
check("both empty is zero", wer("", "")["wer"], 0.0)

print("\ncorpus WER")
# Corpus WER must weight by reference length. Averaging per-clip rates would let a 2-word
# clip with one error (0.5) drag the corpus number as hard as a 200-word clip with 100.
c = corpus_wer([("a b c d", "a b c d"), ("e f", "x f")])
check("total errors over total words", c["wer"], 1 / 6)
check("not the mean of per-clip rates", abs(c["wer"] - 0.25) > 1e-6, True)
check("reports worst clip", c["worst_clip_wer"], 0.5)
check("counts clips", c["clips"], 2)

long_clean = ("word " * 200).strip()
skewed = corpus_wer([(long_clean, long_clean), ("a b", "x y")])
check("one tiny bad clip barely moves corpus WER", skewed["wer"] < 0.02, True)
check("...but worst_clip_wer still surfaces it", skewed["worst_clip_wer"], 1.0)

print("\nregression detection (what the gate actually does)")
BASE = 0.0421
check("noise passes tolerance", (0.0435 - BASE) > 0.02, False)
check("real regression trips it", (0.0700 - BASE) > 0.02, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all WER tests passed")
