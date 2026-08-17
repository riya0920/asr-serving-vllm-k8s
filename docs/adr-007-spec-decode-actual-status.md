# ADR-007: Speculative decoding on Whisper — what actually blocks it

**Status:** INVESTIGATED — partially unblocked, one blocker remains
**Date:** 2026-08-16
**Evidence:** `patches/vllm-0.26-audio-multimodal-spec-decode.patch`, pod logs `spec_test.log`,
`spec3.log`, `spec4.log`

## Correcting an earlier claim in this project

Earlier notes here asserted that vLLM "asserts speculative decoding off for encoder-decoder
models" and that "the feature does not exist for this model class," citing
[vllm#7366](https://github.com/vllm-project/vllm/issues/7366).

**That is wrong for vLLM 0.26.** The claim came from a search summary of an issue filed
against a much older codebase and was never verified against the installed source. Two checks
disproved it:

1. `grep` across the whole vllm package finds **no** line where `speculative` and
   `encoder_decoder` co-occur. There is no such assertion.
2. Launching `vllm-omni serve` on Whisper **with** `--speculative-config` is accepted: vLLM
   builds a `SpeculativeConfig(method='draft_model')`, disables async scheduling for it,
   recomputes `max_num_scheduled_tokens`, and logs

   > "Speculative Decoding with draft models or parallel drafting does not fully support
   > multimodal models yet. Proceeding with tentative support"

So the feature is present and tentatively enabled for this model class. It fails later, for
specific reasons.

## Blocker 1 — fixed here

```
AttributeError: 'WhisperConfig' object has no attribute 'image_token_index'
  vllm/v1/spec_decode/llm_base_proposer.py:1395
```

The proposer assumes every multimodal target is a **vision** model. After a list of known VL
architectures, the `else` branch reads `target_model.config.image_token_index`
unconditionally, then calls `get_language_model()` on the target. Whisper is multimodal by
*audio*: there is no image placeholder token and no separate language submodule.

`patches/vllm-0.26-audio-multimodal-spec-decode.patch` makes both steps conditional and leaves
every vision path untouched. It is small, safe, and upstreamable.

## Blocker 2 — remains

With blocker 1 patched, initialization proceeds further and then fails in the draft model:

```
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not NoneType
```

Reproduced with `--enforce-eager`, so it is a genuine runtime issue rather than a
`torch.compile`/fake-tensor artifact. The draft model's embedding is invoked with `None`
input IDs — vLLM's speculative input plumbing does not populate token IDs on the
encoder-decoder path.

Fixing that means understanding how the proposer builds draft inputs for a model whose
conditioning comes from encoder hidden states rather than a text prefix. That is upstream
engineering measured in days, not a configuration change.

## What this means

The honest statement is **not** "vLLM doesn't support it." It is:

> vLLM 0.26 has tentative draft-model speculative-decoding support that accepts
> encoder-decoder targets. Two bugs block Whisper specifically: a vision-only assumption in
> the proposer (fixed here, patch included) and unpopulated draft input IDs on the
> encoder-decoder path (open).

That is a far more precise position than the incorrect denial it replaces. It is
also a genuine open-source contribution in progress: the patch fixes a real bug that would
affect any audio multimodal target, not just this project.

## Why it would not have moved the throughput number anyway

Worth keeping separate from the above, because it is unchanged by this finding. Speculative
decoding accelerates memory-bound decode steps by exploiting idle compute. ADR-003 measured
the A40 GPU-saturated at 13.07 req/s with continuous batching already filling that idle
compute, and ADR-005/006 found the H100 limited by a serialized per-request path rather than
decode throughput. Spec decoding targets neither constraint.

Where it *would* help is single-request latency at concurrency 1 — which is the regime where
p99 already measures 473 ms.
