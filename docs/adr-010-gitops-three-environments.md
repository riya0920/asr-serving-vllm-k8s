# ADR-010: GitOps promotion across dev, staging and prod

**Status:** MEASURED
**Date:** 2026-08-16

## What ran

Three Argo CD Applications watching three paths in this repository
(`envs/dev`, `envs/staging`, `envs/prod`), each deploying to its own namespace on the k3s
cluster. Nobody ran `kubectl apply` — Argo CD pulled every manifest from GitHub.

| environment | sync policy | config in git | result |
|---|---|---|---|
| dev | automated, selfHeal | 1 replica, 4 slots | **Synced / Healthy** |
| staging | automated, selfHeal | 1 replica, 8 slots | **Synced / Healthy** |
| prod | **none** — human syncs | 2 replicas, 16 slots | **OutOfSync** until promoted, then Synced in **11 s** |

The environments carry genuinely different configuration, so this is three environments rather
than the same manifest applied three times.

The promotion policy is the substance: dev and staging reconcile themselves, prod does not.
A change reaches prod only when a person decides it should. The canary and its automated
analysis still run once that sync begins — the human gate is on *"should this ship"*, which is
judgement, not on *"is it healthy"*, which a machine answers faster and more reliably.

## The blocker

All three Applications sat at `Unknown` indefinitely. The cause was not networking — a test
pod resolved `github.com` and got HTTP 200 — but:

```
InvalidSpecError: Application referencing project default which does not exist
```

Argo CD's `core-install.yaml` does **not** create the `default` AppProject that the full
`install.yaml` ships. Every Application referencing `project: default` is therefore invalid on
a core install, and the symptom is a silent `Unknown` status rather than an obvious error.
Creating the AppProject fixed all three at once.

## Honest scope

- **One cluster, three namespaces** — not three clusters. Namespace-per-environment is common
  in practice and gives separate blast radii and promotion gates without three control planes.
  What it does not give is isolation from a cluster-level failure, which would take all three
  down together.
- The workload is the stub, not Whisper (see [ADR-008](adr-008-keda-autoscaling-measured.md)).

## Scope

Three environments with genuinely different configuration, promoted through Argo CD, with the
single-cluster caveat recorded above.
