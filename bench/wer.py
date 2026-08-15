"""Word error rate — shared by the baseline harness (M2) and the CI gate (M7).

Deliberately dependency-free and deliberately small. The CI gate is the thing standing
between a bad model and production, so it should be something you can read in one sitting
and be sure of, not a wrapper around a package you have not audited.

Normalization matters more than the edit distance here: Whisper emits punctuation and
casing, LibriSpeech references do not. Comparing them raw reports ~40% WER on a perfect
transcript. The normalizer below is the standard "English basic" treatment — lowercase,
strip punctuation, expand a small set of contractions, normalize whitespace. It is NOT
the full Whisper EnglishTextNormalizer (numbers, spelled-out currency, British/American
spelling). That is fine because the gate compares WER against a baseline measured with
THIS SAME normalizer — a consistent, slightly-pessimistic number is all a regression gate
needs. Do not compare these values to published WER benchmarks.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")

_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    for src, dst in _CONTRACTIONS.items():
        text = text.replace(src, dst)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein over words, returning (substitutions, deletions, insertions).

    Two rolling rows instead of a full matrix: the golden set is 50 clips of 30s, which
    is small, but CI runs this on every push and there is no reason to allocate n*m.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m

    # each cell carries (cost, subs, dels, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        curr = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
                continue
            sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
            dele = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
            ins = (curr[j - 1][0] + 1, curr[j - 1][1], curr[j - 1][2], curr[j - 1][3] + 1)
            curr[j] = min(sub, dele, ins, key=lambda c: c[0])
        prev = curr

    _, s, d, i = prev[m]
    return s, d, i


def wer(reference: str, hypothesis: str) -> dict:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    s, d, i = edit_counts(ref, hyp)
    n = len(ref)
    return {
        "wer": (s + d + i) / n if n else (1.0 if hyp else 0.0),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_words": n,
        "hyp_words": len(hyp),
    }


def corpus_wer(pairs: list[tuple[str, str]]) -> dict:
    """Corpus WER = total errors / total reference words.

    NOT the mean of per-clip WERs. Averaging per-clip rates lets one short clip with a
    couple of errors swing the corpus number, which makes the CI gate flaky for reasons
    that have nothing to do with the model.
    """
    tot_s = tot_d = tot_i = tot_n = 0
    per_clip = []
    for ref, hyp in pairs:
        r = wer(ref, hyp)
        per_clip.append(r["wer"])
        tot_s += r["substitutions"]
        tot_d += r["deletions"]
        tot_i += r["insertions"]
        tot_n += r["ref_words"]
    return {
        "wer": (tot_s + tot_d + tot_i) / tot_n if tot_n else 0.0,
        "substitutions": tot_s,
        "deletions": tot_d,
        "insertions": tot_i,
        "ref_words": tot_n,
        "clips": len(pairs),
        "worst_clip_wer": max(per_clip) if per_clip else 0.0,
    }
