# ADR-002: The CUDA 13 wall

**Status:** DECIDED — Option A (rent a CUDA 13 capable host)
**Date:** 2026-08-15

## Context

vLLM v0.20.0 moved its default PyPI wheel and official Docker image to **CUDA 13.0**
([vllm#43435](https://github.com/vllm-project/vllm/issues/43435),
[GPU install docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)). The
RunPod A40 pod used for M2 runs driver **570.195.03**, which caps at CUDA 12.8.

Result: `import vllm` fails with `ImportError: libcudart.so.13`. The compiled extension
`vllm._C_stable_libtorch` links against the CUDA 13 runtime, which cannot load on this driver.

The driver cannot be upgraded from inside a RunPod pod — it is the host's.

## What was tried, and why each failed

| Attempt | Result |
|---|---|
| `pip install vllm==0.26.0` | installs, imports fail — wheel is CUDA 13 |
| `uv pip install --torch-backend=cu128` | **only affects torch.** There is one vllm wheel on PyPI and it is CUDA 13 |
| `--reinstall-package vllm` with cu128 backend | genuinely re-downloaded 289 MB, same CUDA 13 wheel |
| `pip install nvidia-cuda-runtime-cu13` | no wheel for this platform; source build failed |

Note torch itself was already solved separately: the default torch wheel is also CUDA 13 now,
and `--index-url https://download.pytorch.org/whl/cu128` fixes it. torch 2.11.0+cu128 works
fine on this driver. The problem is specific to vLLM's own compiled extension.

## Options

**A. Rent a pod with driver ≥ 580 (CUDA 13 capable).** RunPod exposes a CUDA-version filter
when selecting a host. Everything then installs from PyPI with no workarounds, and the stack
matches the design exactly (vllm 0.26 + vllm-omni 0.26). Keep the same network volume so the
~3 GB of Whisper weights in `/workspace/hf` are not re-downloaded — requires the same region.

**B. Pin vLLM < 0.20.0** (the last CUDA 12 wheels) plus a matching vllm-omni (0.18 / 0.16).
Works on the current pod with no new rental, but freezes the project on an old engine, and
vllm-omni's version-alignment check will complain. Whisper transcription support and the
speculative-decoding internals that M4 has to modify both differ across that gap, so M4 would
be built against a version nobody runs.

**C. CUDA forward compatibility** via `cuda-compat-13-0` plus
`VLLM_ENABLE_CUDA_COMPATIBILITY=1` and `VLLM_CUDA_COMPATIBILITY_PATH`. This is vLLM's own
documented escape hatch and the A40 is a datacenter GPU, so it is supported in principle.
Requires adding NVIDIA's apt repo and installing the compat package. More moving parts, and
the resulting environment is not one a reviewer could reproduce easily.

## Recommendation

**Option A.** It is the only one that leaves the environment matching the design and
reproducible by someone else. B compromises M4, which is the milestone that most needs to be
built against a current engine. C works but produces a bespoke environment.

## Consequence to record either way

This is worth keeping in the write-up. "Rent a GPU and serve a model" hides a real constraint:
as of August 2026 the default wheels for both torch and vLLM require CUDA 13, while a large
share of rentable datacenter GPUs still run 12.x drivers that a tenant cannot upgrade. Two
separate dependency layers had to be pinned to CUDA 12.8 on this box, and one of them —
vLLM's — has no pinnable PyPI wheel at all.
