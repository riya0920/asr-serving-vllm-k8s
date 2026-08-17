# ADR-008: KEDA queue-depth autoscaling, measured on a real cluster

**Status:** MEASURED — autoscaling proven, latency not measurable here
**Date:** 2026-08-16
**Evidence:** `results/m6_wsl_spike_timeline.csv`, `results/m6_wsl_spike.json`

## Substrate

After [ADR-001](adr-001-substrate.md) established that no self-serve RunPod product can host a
Kubernetes node, the cluster went somewhere unexpected: **WSL2 on the laptop**.

```
Ubuntu 26.04 LTS   kernel 6.18.33.2-microsoft-standard-WSL2
systemd = PID 1    cgroup v2    8 cores    3.7 GB RAM
k3s v1.36.3+k3s1   containerd 2.3.2        node Ready in 10s
```

systemd as PID 1 and cgroup v2 — the exact two things every RunPod container lacked. k3s
installed and reached Ready on the first attempt.

**One trap:** Ubuntu 26.04's WSL image ships **without iptables**. k3s logs it as an
informational line at install (`Host iptables-save/iptables-restore tools not found`) and
starts anyway, then pod sandboxes churn endlessly with `SandboxChanged: Pod sandbox changed,
it will be killed and re-created` and containers exit 255. `apt-get install iptables` plus a
k3s restart fixed it permanently. The failure presents as an application crash loop, which
sends you debugging the wrong layer — the stub ran perfectly both locally and inside the
container while the pod was "failing."

## What ran

The project's own manifests, unmodified: `infra/stub/k8s-stub.yaml` and
`infra/keda/scaledobject.yaml` with all three triggers (warm headroom, queue depth, GPU-util
scale-down guard). Real Prometheus scraping the real `vllm:num_requests_waiting` metric — **7
series**, verified before trusting any scaling behaviour.

Per-pod capacity was reduced to 4 slots × 500 ms (8 req/s) so a laptop could saturate a fleet.

## Result

Load: the project's `spike-8x` profile — 4 rps baseline, step to 32 rps, open loop.

| t (s) | desired | ready | queue |
|---|---|---|---|
| 0–42 | 2 | 2 | 0 |
| **44** | 2 | 2 | 2 ← spike arrives |
| **46** | **5** | 2 | 35 |
| 52 | 5 | 5 | 4 |
| 56 | 10 | 5 | 79 |
| 62 | 10 | 10 | 81 |
| 66 | **16** | 10 | 82 |
| 72 | 16 | **16** | drained |

- **Autoscaler reaction: within one 2-second sample.** Queue appeared at t=44; the HPA had
  already raised desired replicas from 2 to 5 by t=46.
- **Scale-out: 2 → 16 replicas** (hit `maxReplicaCount`) in three steps
- **Full fleet ready 28 s after spike onset**, queue drained to zero
- Scale-down held at 16 for the remainder, as designed by
  `stabilizationWindowSeconds: 600`

The tuning that produced this is not accidental: `pollingInterval: 5`, a 5 s scrape interval,
and `stabilizationWindowSeconds: 0` on scale-up. The design notes in the ScaledObject
predicted this behaviour and the measurement matches.

## What this does NOT establish

Recorded plainly, because the temptation to over-claim here is strong.

1. **Latency through the spike is not measurable on this hardware.** The load generator
   flagged its own run invalid: *"generator lagged 2736ms — this run measures the load
   generator, not the server."* A single Python client cannot sustain 32 rps of uploads while
   the same laptop hosts k3s, 16 pods and Prometheus on 8 cores. The p99-under-620ms-through-
   the-spike figure remains unmeasured. The anti-coordinated-omission guard in the load generator is what
   caught this — without it the run would have produced flattering, meaningless numbers.
2. **The pods run the stub, not Whisper.** This proves the autoscaling machinery, not
   "Whisper-Large served on Kubernetes."
3. **Cold start is unrealistically fast.** A single node with a cached 43 MB python image
   starts pods in seconds. Real GPU pods pull multi-GB images and load weights into VRAM —
   minutes. Scale-up latency here is a floor, not a forecast.
4. **Queue depth was sampled from one pod** through a NodePort, so the per-sample values
   oscillate and are not a fleet-wide sum. The trend is sound; individual readings are not.

## Scope

Established: KEDA autoscaling driven by a real `vllm:num_requests_waiting` metric, with real
HPA scaling decisions, absorbing an 8x step increase by scaling 2→16 and draining the queue.

Not established: p99 through the spike, which is not measurable on this hardware.
