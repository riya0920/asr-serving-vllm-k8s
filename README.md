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

| Metric | Baseline | Current | Target | Evidence |
|---|---|---|---|---|
| Throughput (req/s per GPU) | — | — | beat baseline | `results/` |
| p99 latency, 30s clip, steady state | — | — | < 620 ms | `results/` |
| p99 latency through 8x step spike | — | — | < 620 ms | `results/` |
| Speculative decode acceptance rate | n/a | — | measure | `results/` |
| Net spec-decode gain at production batch size | n/a | — | measure | `results/` |
| Cost per audio hour | — | — | beat baseline | `results/` |
| Deploy wall-clock (commit → prod) | — | — | measure | CI run log |

**Baseline** = naive sequential Whisper-Large, one request at a time, same GPU, same audio set.
**Current** = whatever the most recent run in `results/` actually produced.

## Honest status

Nothing is measured yet. See `BUILD_PLAN.md` for milestones; each one must produce a
measurement, not just code.

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
