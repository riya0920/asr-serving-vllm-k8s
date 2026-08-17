# ADR-012: Whisper served on Kubernetes, on a GPU

**Status:** MEASURED
**Date:** 2026-08-17
**Hardware:** Lambda A10 24GB — a real VM with root, unlike every self-serve RunPod product
**Evidence:** `results/m2_whisper_on_k8s_evidence.txt`

## What was proven

A real Whisper-large-v3-turbo model, running in a Kubernetes pod, scheduled onto a GPU by the
NVIDIA device plugin, transcribing correctly.

```
node        170-9-14-117   Ready   v1.36.3+k3s1   containerd 2.3.2
allocatable nvidia.com/gpu: 1
pod         whisper-asr-684d6c7749-v74lj   1/1 Running   10.42.0.9
pod limits  {"nvidia.com/gpu":"1"}         18.2 GB VRAM in use
metrics     vllm:num_requests_waiting / vllm:num_requests_running exposed
```

Output for `clip_000.wav`:

> "Concord returned to its place amidst the tents. The English forwarded to the French baskets
> of flowers, of which they had made a plentiful provision to greet the arrival of the young
> princess..."

which matches the LibriSpeech reference exactly, modulo casing and punctuation.

This closes a real gap in the setup. Previously Whisper ran on rented GPUs
with no Kubernetes, while KEDA autoscaled a stub on a cluster with no GPU — two machines, one
sentence. Both halves now exist on one host, and the metric KEDA scales on is published by the
real engine rather than a simulator.

## Three failures worth keeping

1. **Device plugin: `Failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND`.** k3s only wires the
   NVIDIA runtime into containerd if the toolkit is installed *before* k3s starts. Installing it
   afterwards needs a k3s restart **and** a `RuntimeClass` named `nvidia`, with the device-plugin
   daemonset patched to use it. Without the RuntimeClass the plugin runs under runc and cannot
   see the GPU at all — the node simply reports no `nvidia.com/gpu`.

2. **`Invalid or unsupported audio file` (HTTP 400) from a perfectly healthy pod.** The
   `vllm/vllm-openai` image ships **without** audio dependencies:
   `ImportError('Please install vllm[audio] for audio support')`. The route is registered, the
   health probe is green, the model is loaded — it just cannot decode audio. Installing librosa
   and soundfile at container start fixed it. This is a good example of a readiness probe that
   passes while the service is useless for its actual job.

3. **Single-GPU rolling-update deadlock.** The default `RollingUpdate` strategy creates the new
   pod before terminating the old one. With one GPU on the node, the new pod sits `Pending`
   forever waiting for a resource the old pod holds, and the deployment never converges — while
   `kubectl get pods` still shows a Running pod, so it reads as healthy. `strategy: Recreate` is
   mandatory on single-GPU nodes.

## Honest scope

- **One GPU, so one replica.** KEDA scaling 2 → 16 was measured against the stub fleet
  ([ADR-008](adr-008-keda-autoscaling-measured.md)). Scaling *real* Whisper pods needs either
  multiple GPUs or device-plugin time-slicing — the time-slicing ConfigMap was applied but the
  node still advertised `nvidia.com/gpu: 1` rather than the requested 8, so that remains
  unproven.
- The image is `vllm/vllm-openai`, not `vllm-omni` — the same engine, published as a standard
  image rather than built locally.
