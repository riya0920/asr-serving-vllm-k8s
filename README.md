# ASR Serving — Whisper-Large on vLLM-Omni + Kubernetes autoscaling

Elastic ASR serving: Whisper-Large-v3 served through [vLLM-Omni](https://vllm.ai/blog/vllm-omni)
with continuous batching and shared-encoder speculative decoding, on a Kubernetes fleet that
scales on queue depth via KEDA, delivered through WER-gated CI and Argo CD canary rollouts.

The design question: real ASR traffic spikes 8x at the top of the hour when meetings start.
You cannot provision for peak and idle at 12%. Can the fleet breathe — absorb the spike under
a p99 SLO, shrink back to protect cost — and can changes ship to it in minutes?

## Scoreboard

Every number here starts empty and is filled in **only** from a run artifact in `results/`.
No number in this repo comes from an estimate, a blog post, or a resume draft.

| Metric | Baseline (A40) | Current | Target | Evidence |
|---|---|---|---|---|
| Throughput (req/s per GPU) | **0.632** | — | beat baseline | [M2](results/m2_baseline_sequential.json) |
| Real-time factor | **19.0x** | — | beat baseline | [M2](results/m2_baseline_sequential.json) |
| Latency, 30s clip (p50) | **1594 ms** | — | — | [M2](results/m2_baseline_sequential.json) |
| Latency, 30s clip (max of 50) | **2214 ms** | — | — | [M2](results/m2_baseline_sequential.json) |
| p99 latency through 8x step spike | n/a | — | < 620 ms | pending M6 |
| Corpus WER (CI gate reference) | **0.0160** | — | no regression > 0.02 | [M2](results/m2_baseline_sequential.json) |
| Speculative decode acceptance rate | n/a | — | measure | pending M4 |
| Net spec-decode gain at production batch size | n/a | — | measure | pending M4 |
| Cost per audio hour | — | — | beat baseline | pending M9 |
| Deploy wall-clock (commit → prod) | — | — | measure | pending M8 |

**Baseline** = naive sequential Whisper-Large-v3 (fp16), one request at a time, HF transformers.
**Current** = whatever the most recent run in `results/` actually produced.

### Read the hardware line before quoting any of this

Baseline measured on an **NVIDIA A40 48GB**, not an H100 — RunPod pods cannot host Kubernetes
(see [ADR-001](docs/adr-001-substrate.md)), so the fleet work moved elsewhere and the model
work stayed here. Every *ratio* in this project is therefore honest: M3 and M4 are measured
against this same baseline on this same card. No *absolute* per-GPU throughput number here is
an H100 number, and none should be quoted as one.

No p99 is published for the baseline. 50 samples cannot measure a 99th percentile — nearest
rank would just return the maximum — so the max is reported as the max.

## Honest status

**M1 done.** RunPod standard pods cannot host a Kubernetes node: no `CAP_SYS_ADMIN`,
`ip_forward` not writable, PID 1 is not systemd. Substrate split per ADR-001.

**M2 done.** Golden set: 50 clips, each exactly 30s, whole utterances only, silence-padded,
81% speech density. Sequential baseline: 0.632 req/s, 19.0x real-time, WER 0.0160.

Everything else is unmeasured. See `BUILD_PLAN.md`.

### A bug worth recording

The first M2 run reported WER **0.1504**. The model was fine — the golden set was not.
`build_golden.py` filled its buffer past 30s, cut the audio at exactly 30s, and dropped the
whole trailing utterance from the reference, leaving unreferenced speech in every clip.
Whisper transcribed it correctly and every word scored as an insertion.

The error breakdown is what gave it away: 45 substitutions but 277 deletions and 224
insertions. Substitutions are what a model gets *wrong*; deletions and insertions at that
ratio are an alignment problem. After the fix, substitutions stayed at 45 while deletions
fell to 5 and insertions to 7.

Had 0.1504 been committed as the CI reference, the gate would have permitted a model to
degrade to ~15% WER and still pass — the exact silent failure the gate exists to catch.

Two known risks are tracked openly rather than buried:

1. **Speculative decoding is not supported for encoder-decoder models in vLLM today**
   ([vllm#7366](https://github.com/vllm-project/vllm/issues/7366) — the enc-dec path asserts it
   off). M4 implements the narrow shared-encoder case. Until M4 lands and reports an acceptance
   rate, this project does not claim speculative decoding.
2. **Speculative decoding and continuous batching compete for the same resource.** Spec decode
   pays off because decode steps leave compute idle; continuous batching exists to fill that
   idle compute with other requests. The gain measured at batch=1 will not hold at batch=64.
   Every spec-decode number in this repo is reported at production batch size, and the batch=1
   number is reported next to it so the gap is visible.

## Layout

```
bench/       baseline harness + step-function load generator
scripts/     environment probes and setup
infra/       k8s manifests, KEDA ScaledObject, Argo CD apps  (M5+)
ci/          GitHub Actions workflows, WER gate              (M7+)
golden/      golden audio clips + reference transcripts      (M2)
results/     raw run artifacts — the only source of numbers
docs/        design notes and decision records
```
