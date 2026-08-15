#!/usr/bin/env python3
"""Catch the manifest mistakes that deploy cleanly and then quietly do nothing.

`kubectl apply` accepts all of these. The cluster reports Healthy. The failure only shows up
as "the autoscaler never scaled" during the one expensive benchmark session you rented the
hardware for. Every check here exists because it is invisible at apply time:

  - a ServiceMonitor naming a port the Service does not expose -> no scrape -> no metric ->
    KEDA queries an empty result and holds at minReplicaCount, looking perfectly healthy
  - a Service selector that misses the pod labels -> endpoints empty -> same silence
  - a `replicas:` field on a KEDA-managed Deployment -> every Argo sync fights the HPA
  - a PDB minAvailable above the autoscaler's floor -> node drains block forever
  - scrape interval slower than KEDA's polling interval -> scaling on stale numbers

Stdlib + pyyaml only; runs in the cheapest CI job.

    python ci/validate_manifests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

ROOT = Path(__file__).parent.parent
ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def load_all() -> list[tuple[Path, dict]]:
    docs: list[tuple[Path, dict]] = []
    for path in sorted(list(ROOT.glob("infra/**/*.yaml")) + list(ROOT.glob(".github/**/*.yml"))):
        try:
            for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
                if isinstance(doc, dict):
                    docs.append((path, doc))
        except yaml.YAMLError as e:
            err(f"{path.relative_to(ROOT)}: does not parse: {e}")
    return docs


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def main() -> int:
    docs = load_all()
    print(f"parsed {len(docs)} documents from {len({p for p, _ in docs})} files")

    by_kind: dict[str, list[tuple[Path, dict]]] = {}
    for p, d in docs:
        by_kind.setdefault(d.get("kind", "?"), []).append((p, d))

    # ---------------------------------------------------------------- workloads and labels
    workloads: dict[str, dict] = {}
    for kind in ("Deployment", "Rollout", "StatefulSet"):
        for p, d in by_kind.get(kind, []):
            name = d["metadata"]["name"]
            workloads[name] = d

            spec = d.get("spec", {})
            if kind == "Deployment" and "replicas" in spec:
                err(f"{rel(p)}: Deployment/{name} sets spec.replicas while KEDA manages "
                    f"replica count — every sync will fight the autoscaler")

            pod_labels = (spec.get("template", {}).get("metadata", {}).get("labels")) or {}
            sel = (spec.get("selector", {}) or {}).get("matchLabels", {}) or {}
            if pod_labels and sel and not set(sel.items()) <= set(pod_labels.items()):
                err(f"{rel(p)}: {kind}/{name} selector {sel} does not match pod labels "
                    f"{pod_labels} — the workload will never own its pods")

    # ---------------------------------------------------------------- services and ports
    services: dict[str, dict] = {}
    for p, d in by_kind.get("Service", []):
        name = d["metadata"]["name"]
        services[name] = d
        ports = d.get("spec", {}).get("ports", [])
        if any("name" not in x for x in ports):
            warn(f"{rel(p)}: Service/{name} has an unnamed port — ServiceMonitor and "
                 f"Rollout analysis both select ports by name")

    # ---------------------------------------------------------------- scrape wiring
    for p, d in by_kind.get("ServiceMonitor", []):
        name = d["metadata"]["name"]
        sel = d.get("spec", {}).get("selector", {}).get("matchLabels", {})

        matched = [
            s for s in services.values()
            if set(sel.items()) <= set((s.get("metadata", {}).get("labels") or {}).items())
        ]
        if not matched:
            err(f"{rel(p)}: ServiceMonitor/{name} selector {sel} matches no Service in this "
                f"repo — nothing will be scraped and KEDA will scale on an empty query")

        for ep in d.get("spec", {}).get("endpoints", []):
            port = ep.get("port")
            for svc in matched:
                names = {x.get("name") for x in svc.get("spec", {}).get("ports", [])}
                if port not in names:
                    err(f"{rel(p)}: ServiceMonitor/{name} scrapes port '{port}' but "
                        f"Service/{svc['metadata']['name']} exposes {sorted(n for n in names if n)}")

    # ---------------------------------------------------------------- KEDA
    for p, d in by_kind.get("ScaledObject", []):
        name = d["metadata"]["name"]
        spec = d.get("spec", {})
        target = spec.get("scaleTargetRef", {}).get("name")
        if target and target not in workloads:
            err(f"{rel(p)}: ScaledObject/{name} targets '{target}' which is not defined "
                f"in this repo")

        lo = spec.get("minReplicaCount", 0)
        hi = spec.get("maxReplicaCount", 100)
        if lo > hi:
            err(f"{rel(p)}: ScaledObject/{name} minReplicaCount {lo} > maxReplicaCount {hi}")
        if lo == 0:
            warn(f"{rel(p)}: ScaledObject/{name} can scale to zero — the first request after "
                 f"an idle period pays the full GPU cold start and will time out")

        triggers = spec.get("triggers", [])
        if not triggers:
            err(f"{rel(p)}: ScaledObject/{name} has no triggers")
        if not any("waiting" in str(t.get("metadata", {}).get("query", "")) for t in triggers):
            err(f"{rel(p)}: ScaledObject/{name} has no queue-depth trigger — scaling on "
                f"utilization alone cannot tell 'busy' from 'overloaded'")

        polling = spec.get("pollingInterval", 30)
        for _, sm in by_kind.get("ServiceMonitor", []):
            for ep in sm.get("spec", {}).get("endpoints", []):
                iv = str(ep.get("interval", "30s")).rstrip("s")
                if iv.isdigit() and int(iv) > polling:
                    err(f"{rel(p)}: scrape interval {iv}s is slower than KEDA "
                        f"pollingInterval {polling}s — KEDA will decide on stale metrics")

        # PDB floor must not exceed the autoscaler floor, or drains deadlock.
        for pp, pdb in by_kind.get("PodDisruptionBudget", []):
            ma = pdb.get("spec", {}).get("minAvailable")
            if isinstance(ma, int) and ma > lo:
                err(f"{rel(pp)}: PDB minAvailable {ma} exceeds ScaledObject minReplicaCount "
                    f"{lo} — node drains will block indefinitely at minimum scale")

    # ---------------------------------------------------------------- probes
    for kind in ("Deployment", "Rollout"):
        for p, d in by_kind.get(kind, []):
            for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
                if not c.get("readinessProbe"):
                    err(f"{rel(p)}: container '{c.get('name')}' has no readinessProbe — "
                        f"scale-up will route traffic to pods still loading weights, so a "
                        f"spike shows up as a burst of errors")
                sp, lp = c.get("startupProbe"), c.get("livenessProbe")
                if lp and not sp and c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu"):
                    err(f"{rel(p)}: GPU container '{c.get('name')}' has a livenessProbe but "
                        f"no startupProbe — the probe will kill the pod during the multi-minute "
                        f"model load and crash-loop forever")

    # ---------------------------------------------------------------- report
    print()
    for w in WARNINGS:
        print(f"WARN  {w}")
    for e in ERRORS:
        print(f"ERROR {e}")
    print()
    if ERRORS:
        print(f"{len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
        return 1
    print(f"manifests OK ({len(WARNINGS)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
