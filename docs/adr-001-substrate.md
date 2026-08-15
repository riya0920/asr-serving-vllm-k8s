# ADR-001: Where the Kubernetes half runs

**Status:** OPEN — blocked on the M1 probe
**Date opened:** 2026-08-15

## Context

The project needs two different things from its hardware:

- **Model work (M2–M4)** — a GPU and a Python environment. Any container will do.
- **Cluster work (M5–M8)** — a real Kubernetes *node*: kubelet managing cgroups, mount
  propagation, the NVIDIA device plugin claiming GPUs as allocatable resources.

RunPod is already available, so it is the default candidate. The problem is that a standard
RunPod Pod is a container, not a host. k3s inside a container needs `--privileged` for cgroup
and mount access, and the device plugin expects to configure the node's container runtime.
RunPod's own documentation points at Bare Metal for "full system-level access", which suggests
standard Pods will not do it — but their capability set changes and the docs are not a
substitute for trying it.

## Decision

Not yet made. `scripts/runpod_probe.sh` decides it empirically. It costs a few cents on the
cheapest GPU pod and tests the exact thing that matters: does `k3s kubectl get nodes` report
Ready, and does the node advertise `nvidia.com/gpu`.

## Options

| Option | Model work | Cluster work | Cost | Notes |
|---|---|---|---|---|
| A — RunPod Pods only | yes | **only if probe passes** | lowest | test first |
| B — RunPod Pods + RunPod Bare Metal | yes | yes | higher, reserved | full root, no container layer |
| C — RunPod Pods + root-access VM (e.g. Lambda) | yes | yes | mid, on-demand | k3s installs cleanly on a real VM |
| D — RunPod for GPU, local kind cluster for control plane | yes | partial | lowest | KEDA/Argo logic provable, but no GPU nodes — cannot produce the spike number |

Option D is a legitimate fallback for *developing* M5–M8 cheaply, and should be used for that
regardless of which option wins: write and test every manifest against a CPU cluster with a
stub server that exposes a fake `num_requests_waiting`, so the expensive box only ever runs
the measurement.

## Consequences to record once decided

- Which option, and the probe output that justified it (`results/m1_substrate_probe.txt`)
- Hourly cost and what a full benchmark session is expected to cost
- If the cluster ends up on a single multi-GPU node: note that image pull is near-zero there,
  so the cold-start story is dominated by model-load-to-VRAM. Say so in the write-up rather
  than implying a multi-minute pull that was never on the critical path.
