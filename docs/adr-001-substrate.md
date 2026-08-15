# ADR-001: Where the Kubernetes half runs

**Status:** DECIDED — Option C (split substrate)
**Date:** 2026-08-15
**Evidence:** `results/m1_substrate_probe.txt`

## Context

The project needs two different things from its hardware:

- **Model work (M2–M4)** — a GPU and a Python environment. Any container will do.
- **Cluster work (M5–M8)** — a real Kubernetes *node*: kubelet managing cgroups, mount
  propagation, the NVIDIA device plugin claiming GPUs as allocatable resources.

RunPod was the default candidate because it was already available.

## What the probe found

Ran on a RunPod GPU pod (NVIDIA A40 48GB, Ubuntu 22.04, kernel 6.8.0-65, container
`9a8af355e97c`). Three independent blockers, each sufficient on its own:

1. **`CAP_SYS_ADMIN` is not granted.** The container holds only the default Docker set
   (`chown, dac_override, fowner, fsetid, kill, setgid, setuid, setpcap, net_bind_service,
   net_raw, sys_chroot, mknod, audit_write, setfcap`). Kubelet cannot manage mounts or
   cgroups without it.
2. **`/proc/sys/net/ipv4/ip_forward` is not writable**, and neither `br_netfilter` nor
   `overlay` can be loaded. Pod networking cannot be configured.
3. **PID 1 is not systemd.** The k3s installer completed and wrote its unit file, then died
   at `System has not been booted with systemd as init system (PID 1). Can't operate.`

Note that (3) alone could be worked around by running `k3s server` as a bare process instead
of a service — but (1) and (2) cannot be worked around from inside the container, so it is
not worth attempting.

Incidental finding: the probe reported 96 CPUs and 503 GB RAM, which is the **host's**
`/proc`, not the pod's allocation. Any resource-based tuning done inside a RunPod pod is
reading the wrong numbers.

## Decision

**Option C — split the substrate.**

| Milestone | Where | Why |
|---|---|---|
| M2 baseline, M3 vLLM-Omni, M4 spec decode | RunPod GPU pod | container is fine; A40 48GB is ample for Whisper-Large + a Distil-Whisper draft |
| M5/M6/M8 control-plane rehearsal | any VM with Docker (kind) | proves the whole KEDA→HPA→pod loop against `infra/stub/`, no GPU needed, effectively free |
| M6 real spike numbers on a GPU fleet | deferred — needs a root-access GPU VM | RunPod Bare Metal, or a provider that gives real VMs |

The rehearsal target should be the **Oracle free-tier VM already in use for job-hunter**
(4 OCPU / 24 GB ARM). `kind`, KEDA, Prometheus and `python:3.12-slim` all have arm64 images,
so the entire autoscaling control plane can be validated at zero additional cost. Only the
final GPU-fleet measurement needs paid hardware.

## Consequences

- The 8x spike number cannot be produced on RunPod standard pods. It is deferred to a
  single, well-prepared session on root-access GPU hardware, with every manifest already
  validated against the kind rehearsal.
- The baseline (M2) and all single-GPU throughput work happen on an **A40**, not an H100.
  Every speedup is measured against a baseline on the same A40, which is what makes the
  ratio meaningful. Absolute per-GPU throughput will not be an H100 number and must not be
  reported as one.
- Cold-start measurements taken later on a single multi-GPU node will under-represent image
  pull time, since the image is node-local. Report the model-load-to-VRAM component
  separately rather than implying a multi-minute pull.
