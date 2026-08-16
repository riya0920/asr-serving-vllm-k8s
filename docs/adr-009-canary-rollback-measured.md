# ADR-009: Argo Rollouts canary with analysis-driven auto-rollback

**Status:** MEASURED
**Date:** 2026-08-16
**Cluster:** k3s v1.36.3 on WSL2 (see [ADR-008](adr-008-keda-autoscaling-measured.md))

## What was installed

- **Argo Rollouts** — the canary controller
- **Argo CD core** — application-controller, repo-server, redis, applicationset-controller

Both healthy alongside Prometheus, KEDA and the stub fleet on 3.7 GB of RAM.

## The demonstration

A `Rollout` with canary steps 25% → pause → 50% → pause → 100%, gated by an `AnalysisTemplate`
that queries Prometheus and evaluates `result[0] >= {{args.floor}}`. Two runs, identical
except for the threshold.

| run | floor | AnalysisRun | measurements | rollout | time |
|---|---|---|---|---|---|
| good | `0` | **Successful** | `Successful([0])` ×3 | **Healthy** | 74 s |
| bad | `999999999` | **Failed** | `Failed([0])` ×2 | **Degraded — rolled back** | 22 s |

Same query, same returned value, opposite verdicts. The gate passes on merit and fails on
merit, which is the only property that makes it a gate. The bad revision's ReplicaSet went to
zero while the previous revision retained its replicas.

## A false positive I nearly published

The first attempt *looked* like a success — good canary promoted, bad canary rolled back, both
with plausible timings. It was wrong.

Both AnalysisRuns showed `phase=Error`, and the measurements said:

```
Post "http://10.42.0.136:9090/api/v1/query": dial tcp 10.42.0.136:9090:
connect: no route to host
```

The AnalysisTemplate had Prometheus's **pod IP** baked in, captured when the template was
written. The pod restarted, the IP moved to `10.42.0.174`, and every measurement failed. Five
consecutive errors exceeded Argo's `consecutiveErrorLimit`, which aborts a rollout — so the
"auto-rollback" was **error-driven, not analysis-driven**. The observable outcome was identical
to success.

Two lessons, both worth more than the demo itself:

1. **Never address an in-cluster service by pod IP.** Use the service DNS name
   (`prometheus-operated.monitoring.svc:9090`). Pod IPs are ephemeral by design.
2. **A rollback is not evidence that a gate works.** Argo aborts on failed analysis *and* on
   errored analysis, and the rollout phase looks the same either way. Read the AnalysisRun's
   own phase and its per-measurement values — `Failed([0])` versus `Error(no route to host)` is
   the difference between a working gate and a broken one that happens to abort.

The corrected run shows `Successful([0])` and `Failed([0])`: a real number, fetched over
service DNS, evaluated against a real condition.

## Claims

- **#14 Argo CD canary rollouts — met** for the canary mechanism: progressive traffic steps,
  automated Prometheus analysis, and auto-rollback on breach, all measured.
- Not demonstrated: promotion across **separate dev/staging/prod clusters**. One k3s node hosts
  a single environment. The `infra/argocd/applications.yaml` manifests define all three with
  automated sync for dev/staging and manual for prod, but only one was exercised.
