# Build plan

Rule carried over from the TTS project: **every milestone produces a measurement, not just
code.** A milestone is not done when it runs; it is done when there is a JSON file in
`results/` and a scoreboard row updated from it.

Second rule, specific to this project: **the expensive box only runs measurements.** Develop on
the cheapest GPU that fits the model, rent the big hardware for benchmark sessions, stop it the
moment the run finishes. It bills while idle.

---

## M0 — Repo skeleton, load generator, honesty rules ✅

README with an empty scoreboard, this plan, `bench/loadgen.py`, `scripts/runpod_probe.sh`.
No GPU required. Done.

## M1 — Decide the infrastructure substrate

Run `scripts/runpod_probe.sh` on the cheapest RunPod GPU pod. It answers one question: **can a
real Kubernetes node exist here?**

- **Probe passes** (k3s installs, node Ready, device plugin sees the GPU) → everything runs on
  RunPod. Cheapest path.
- **Probe fails** (no CAP_SYS_ADMIN / cgroup writes blocked / kubelet won't start) → split:
  RunPod Pods for all model work (M2–M4), and either RunPod Bare Metal or a root-access VM
  provider for the cluster work (M5–M6). Record the decision in `docs/adr-001-substrate.md`.

Do not start M5 before this is answered. The whole infra half depends on it.

**Artifact:** `results/m1_substrate_probe.txt`, `docs/adr-001-substrate.md`

## M2 — Golden audio set + sequential baseline

Assemble ~50 clips of 30s with reference transcripts (LibriSpeech test-clean concatenated to
30s is the honest default — real speech, known references, redistributable). These serve double
duty: the benchmark corpus and the CI WER gate corpus.

Then run `bench/baseline_sequential.py`: Whisper-Large-v3 through HF transformers, one request
at a time. This is the denominator for every speedup claim in the project.

Also record baseline WER on the golden set. Every later change is measured against it.

**Artifact:** `results/m2_baseline_sequential.json`, `golden/manifest.json`

## M3 — vLLM-Omni serving with continuous batching

Whisper-Large-v3 behind vLLM-Omni's OpenAI-compatible transcription endpoint. Sweep
concurrency; find the batch size where p99 crosses 620ms. **Isolate this number** — it is the
continuous-batching contribution, measured before any spec-decode work exists, and the
project's story depends on knowing the split.

Confirm `vllm:num_requests_waiting` is exposed on `/metrics`. M5 scales on it.

**Artifact:** `results/m3_vllm_batching.json` (throughput + p50/p95/p99 vs concurrency)

## M4 — Shared-encoder speculative decoding  ⚠ long pole

vLLM asserts spec decode off for encoder-decoder models. Implement the narrow case:

Distil-Whisper copies Whisper-Large's encoder **verbatim** and keeps 2 of 32 decoder layers.
So the encoder runs once and both decoders consume the same encoder hidden states and the same
cross-attention KV. This is not general enc-dec speculative decoding — it is the shared-encoder
special case, which is a much smaller change.

Steps: reproduce the assertion → build a standalone shared-encoder draft/verify loop outside
the scheduler → measure acceptance rate on the golden set → integrate with the batching
scheduler → measure again under load.

**Report three numbers, always together:** acceptance rate, gain at batch=1, gain at production
batch size. If the third is near zero, that is the finding, and it is a more interesting thing
to be able to explain than a speedup.

**Artifact:** `results/m4_spec_decode.json`, `docs/adr-002-spec-decode.md`

## M5 — Kubernetes + KEDA on queue depth

One pod per GPU via the NVIDIA device plugin. Prometheus scrapes vLLM. KEDA ScaledObject scales
primarily on **queue depth per pod** — the leading, unbounded signal — with DCGM GPU
utilization only as a scale-down guard (util saturates at ~95% and cannot distinguish "busy"
from "drowning", so it must never be the scale-up trigger).

Tune the target so steady-state pods sit at ~70% of capacity; that headroom is what absorbs the
first seconds of a spike while new pods start. Aggressive scale-up steps, long scale-down
stabilization window — flapping a pod down 30s before the next spike is worse than briefly
over-provisioning.

Measure pod cold start and break it down: schedule → image pull → model load to VRAM. Note
honestly that on a single multi-GPU node the image is already local, so the pull component is
near zero and model-load dominates.

**Artifact:** `results/m5_cold_start.json`, `infra/keda/scaledobject.yaml`

## M6 — The adversarial spike run

`bench/loadgen.py --profile spike-8x`. A **step function, not a ramp** — traffic jumps 8x within
seconds, the worst case for autoscaler reaction time. Record client-side p99 across the spike
window, queue depth over time, pod count over time, and time-to-recovery.

This is the run that earns the headline number or doesn't. Either outcome gets published to the
scoreboard.

**Artifact:** `results/m6_spike_8x.json` + timeseries CSV

## M7 — CI gates

GitHub Actions, three gates on every change:

1. **Model artifact validation** — checksum weights against a manifest, load the model, run the
   golden set, assert WER within tolerance of the recorded baseline. This catches the silent
   killer: a preprocessing config typo that crashes nothing and just transcribes worse.
2. **Load test** — short run against an ephemeral environment, assert throughput and p99 have
   not regressed. An accurate deployment that is 40% slower is also a failed deployment.
3. **Build, scan, push** with an immutable tag.

**Artifact:** green CI run + a deliberately-broken PR proving the WER gate fails it

## M8 — GitOps delivery with canary

Config repo holds desired state for dev/staging/prod. Argo CD reconciles. Argo Rollouts does the
canary: small traffic slice → automated Prometheus analysis on p99 and error rate over a soak →
advance or auto-rollback. Nobody runs `kubectl apply` at prod, including me. A deploy is a PR.

Record real wall-clock: commit → prod. Break it down by stage so the number is defensible.

**Artifact:** `results/m8_deploy_timing.json`, a real auto-rollback captured in a run log

## M9 — Cost model

Cost per audio hour, baseline vs current, derived from measured throughput and the actual
rented price. Include the warm headroom and the always-on floor explicitly — the savings will
be **lower** than throughput gain alone predicts, and knowing why is the point.

**Artifact:** `results/m9_cost_model.json` with the arithmetic shown

---

## Sequencing note

M2 → M3 → M4 are model work and can all run on one cheap GPU. M5 → M6 need the fleet and are
the only milestones that need real money. Batch them into one benchmark session: have every
manifest written, tested against a CPU/tiny-model cluster, and committed **before** the big box
is started.
