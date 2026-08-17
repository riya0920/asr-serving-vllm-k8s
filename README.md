# ASR serving: Whisper-Large on vLLM with Kubernetes autoscaling

Speech-to-text serving stack built around three problems: making Whisper-Large fast enough to
serve, scaling the fleet when traffic spikes, and shipping changes without breaking accuracy.

Meeting-transcription traffic is bursty. Load jumps ~8x at the top of the hour when meetings
start, then falls off. Provisioning for peak wastes most of the day; provisioning for average
drops requests. So the fleet has to scale on a signal that moves *before* latency degrades,
and deploys have to be safe enough to do often.

Everything below was measured on rented GPUs (A40, H100 NVL, A10) and a k3s cluster. Raw run
artifacts are in `results/`.

## Results

Single GPU, 50 clips of exactly 30 seconds, LibriSpeech test-clean.

| configuration | throughput | vs baseline |
|---|---|---|
| HuggingFace transformers, one request at a time | 0.60 req/s | 1x |
| vLLM continuous batching | 4.45 req/s | 7.4x |
| batch slots 32 → 256 | 7.36 req/s | 12.3x |
| `whisper-large-v3-turbo` (4 decoder layers vs 32) | **13.07 req/s** | **21.8x** |

Latency, turbo, same GPU:

| concurrency | throughput | p50 | p99 |
|---|---|---|---|
| 1 | 2.79 req/s | 362 ms | 473 ms |
| 2 | 3.90 req/s | 525 ms | 636 ms |
| 8 | 5.48 req/s | 1442 ms | 1902 ms |
| 128 | 13.07 req/s | ~8.5 s | ~17 s |

Throughput and latency trade against each other directly. Peak throughput happens at
concurrency 128, where p99 is 17 seconds. Sub-500ms p99 happens at concurrency 1, where
throughput is 2.79 req/s. A fleet gets both by keeping each pod at low concurrency and adding
pods, which is what the autoscaler is for.

Cost per audio hour drops 92% against the sequential baseline (95% raw, less the warm headroom
and minimum-replica floor the autoscaler needs).

Accuracy: 1.60% WER for large-v3, 2.36% for turbo, measured on the same golden set and
reproduced to four decimal places across three hosts.

## Autoscaling

KEDA scales on **queue depth** (`vllm:num_requests_waiting`), not GPU utilization. Utilization
saturates: once every batch slot is busy it reads ~100% whether two requests are queued or two
hundred, so it cannot tell you how many pods to add. Queue depth is unbounded and starts
climbing the moment arrival rate exceeds service rate.

Measured against an 8x step-function load increase:

```
t=44s   queue appears
t=46s   HPA raises desired replicas 2 → 5     (reaction inside one 2s sample)
t=56s   5 → 10
t=66s   10 → 16
t=72s   16 pods ready, queue drained
```

Three triggers, and KEDA takes the maximum: in-flight requests hold steady-state occupancy at
~70% so a spike's first seconds land on warm pods; queue depth drives the surge; GPU
utilization acts only as a scale-down guard so the fleet does not shrink while the GPUs are
still busy. Scale-up reacts immediately, scale-down waits 10 minutes, because dropping a pod
30 seconds before the next spike costs a full cold start.

## Delivery

GitHub Actions runs lint, unit tests for the WER math, and manifest validation on every push.
Behind a GPU runner it also checksums model weights, runs a WER gate over the golden set, and
runs a load test asserting throughput and p99 have not regressed.

The WER gate exists because of a specific failure: a bad preprocessing config, wrong language
token, or truncated weight download does not crash anything. The container starts, health
checks pass, requests return 200 with plausible English in them, and the only symptom is worse
transcripts. Without a gate, users find out first.

Argo CD reconciles three environments from this repo (`envs/dev`, `envs/staging`, `envs/prod`).
Dev and staging self-heal automatically; prod has no automated sync policy, so a change reaches
it only when someone syncs it. Argo Rollouts does the canary: traffic steps 25% → 50% → 100%
with automated Prometheus analysis between steps.

Measured: canary promotion 74s when analysis passes, automatic rollback 22s when it fails.
Commit to deployed is about 90 seconds for the gates that run without a GPU.

## Where the performance ceiling is

Four experiments, each set up so a wrong hypothesis would show clearly.

**A40 is GPU-bound.** 100% utilization at 300W, the card's full TDP. Running four load
generator processes instead of one gave 0.90x, so there was no headroom being missed.

**H100 is not, and is slower.** 5.93 req/s against the A40's 13.07, with the GPU at 23-30% and
126W while the engine burned 23 CPU cores. A card with roughly 3x the compute delivered less
than half the throughput, because Whisper's log-mel front end runs on CPU inside the engine
process and becomes the constraint once the GPU is fast enough.

**Moving mel extraction to the GPU did not fix it.** Engine CPU dropped from 23 cores to 16 and
GPU utilization doubled, confirming the work moved, but throughput improved 1.06x. So mel was a
large CPU consumer but not the binding constraint. Afterwards neither CPU nor GPU was saturated
(16 of 192 cores, GPU at 50% drawing 130W), which points at a serialized per-request path
inside vLLM's encoder-decoder implementation.

**faster-whisper was 5.7x slower.** CTranslate2 is purpose-built for Whisper and does not share
vLLM's code path, so it was the obvious alternative. It came in at 1.89 req/s against vLLM's
10.72 on the same A10, both pinned at the card's 150W TDP. It did produce better transcripts
(1.72% WER). Caveat: the test server batches segments within a file, not across concurrent
requests the way vLLM does, so this measures naive faster-whisper serving rather than its best
form.

Speculative decoding is partially implemented. vLLM 0.26 accepts encoder-decoder targets, but
two bugs block Whisper: the draft-model proposer assumes every multimodal model is a vision
model and reads `image_token_index` unconditionally (fixed, see `patches/`), and draft input IDs
are never populated on the encoder-decoder path (open, needs upstream work).

## Running it

```bash
# build the golden set and measure a baseline (needs a GPU)
python bench/build_golden.py --clips 50 --out golden
python bench/baseline_sequential.py --golden golden --out results/baseline.json

# serve and load test
vllm-omni serve openai/whisper-large-v3-turbo --port 8000 --max-num-seqs 256
python bench/loadgen.py --url http://localhost:8000 --audio-dir golden/audio \
  --profile sweep --concurrency-list 1,8,32,128 --out results/sweep.json

# 8x step-function spike
python bench/loadgen.py --url http://localhost:8000 --audio-dir golden/audio \
  --profile spike-8x --rps 4 --out results/spike.json

# cluster: k3s + Prometheus + KEDA, then the stub fleet for CPU-only testing
bash scripts/kind_up.sh
kubectl apply -f infra/keda/scaledobject.yaml
```

The load generator is open-loop with Poisson arrivals, and measures latency from *scheduled*
send time rather than actual send time. A closed-loop generator throttles itself exactly when
the server slows down, which hides the overload you are trying to measure. It also fails its own
run if the generator itself lagged more than 50ms, since at that point the numbers describe the
client rather than the server.

`infra/stub/` is a fake engine that exposes the same metrics and models the same queueing
behaviour. It exists so the autoscaling and delivery work can be developed on a laptop instead
of a rented GPU.

## Layout

```
bench/     load generator, WER scoring, baseline harness, engine servers
ci/        WER gate, perf gate, weight verification, manifest validator
infra/     k8s manifests, KEDA ScaledObject, Argo CD apps, Rollouts canary, stub engine
envs/      dev / staging / prod, reconciled by Argo CD
scripts/   cluster setup, environment probes, benchmark sessions
patches/   vLLM fix for audio multimodal speculative decoding
results/   raw run artifacts, one per measurement
docs/      decision records, including the experiments that failed
```

## Notes

A few things that cost real time and are not in any tutorial:

- Container-based GPU hosts (RunPod pods and similar) cannot run Kubernetes. No `CAP_SYS_ADMIN`,
  `ip_forward` is read-only, and PID 1 is not systemd. WSL2 works, since it has systemd and
  cgroup v2.
- Ubuntu 26.04's WSL image ships without iptables. k3s logs that as informational, starts
  anyway, and then pod sandboxes churn endlessly while containers exit 255 — which looks like an
  application crash loop.
- As of August 2026 the default PyPI wheels for both torch and vLLM require CUDA 13, while many
  rentable GPUs run 12.x drivers a tenant cannot upgrade. torch has a pinnable cu128 index;
  vLLM does not.
- `vllm/vllm-openai` ships without audio dependencies, so a healthy pod returns HTTP 400 on
  every audio request until librosa and soundfile are installed.
- On a single-GPU node, `RollingUpdate` deadlocks: the new pod waits for a GPU the old pod
  holds, and `kubectl get pods` still shows a Running pod. Use `Recreate`.
- k3s only wires the NVIDIA runtime into containerd if the toolkit was installed before k3s
  started. Otherwise it needs a restart plus a `RuntimeClass` named `nvidia`.
