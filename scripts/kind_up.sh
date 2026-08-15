#!/usr/bin/env bash
# Stand up the ENTIRE autoscaling control plane on a laptop-class box, with zero GPUs.
#
# kind cluster + Prometheus + KEDA + the stub ASR server + the real ScaledObject. Everything
# except the model is identical to production, so every mistake that can be found without a
# GPU gets found here, for free, instead of at $25/hour on the rented multi-GPU box.
#
#   bash scripts/kind_up.sh
#   bash scripts/kind_test_autoscale.sh     # drive it and watch replicas move
#   kind delete cluster --name asr
#
# Requires: docker, kind, kubectl, helm. NOT runnable on the Windows laptop (no Docker) —
# run it on any cheap CPU VM or the RunPod pod.

set -euo pipefail

CLUSTER=asr
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> creating kind cluster '$CLUSTER'"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  cat <<'EOF' | kind create cluster --name asr --config -
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF
else
  echo "    already exists"
fi
kubectl config use-context "kind-${CLUSTER}"

echo "==> installing Prometheus (kube-prometheus-stack)"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.enabled=false \
  --set alertmanager.enabled=false \
  --set prometheus.prometheusSpec.scrapeInterval=5s \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --wait --timeout 10m

echo "==> installing KEDA"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace --wait --timeout 5m

echo "==> deploying the stub ASR server"
kubectl create namespace asr --dry-run=client -o yaml | kubectl apply -f -
# Ship the stub as a ConfigMap rather than building an image: one source of truth
# (infra/stub/stub_asr_server.py), no registry, no build step.
kubectl -n asr create configmap stub-src \
  --from-file=stub_asr_server.py="$ROOT/infra/stub/stub_asr_server.py" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/infra/stub/k8s-stub.yaml"

echo "==> waiting for the stub to be scraped by Prometheus"
kubectl -n asr rollout status deploy/whisper-asr --timeout=180s

echo "==> applying the real ScaledObject (with the stub as its target)"
# The ScaledObject is applied unmodified — same triggers, same thresholds, same behavior
# blocks that production uses. Only the workload behind it is fake.
sed 's|prometheus-operated.monitoring.svc:9090|prometheus-operated.monitoring.svc:9090|' \
  "$ROOT/infra/keda/scaledobject.yaml" | kubectl apply -f -

echo
echo "==> up. verify the metric pipeline actually works before trusting any scaling:"
echo
echo "  kubectl -n monitoring port-forward svc/prometheus-operated 9090:9090 &"
echo "  curl -s 'localhost:9090/api/v1/query?query=sum(vllm:num_requests_waiting)' | jq .data.result"
echo
echo "  # ^ if that returns an empty array, KEDA is scaling on nothing and will sit at"
echo "  #   minReplicaCount forever while looking perfectly healthy. Check this FIRST."
echo
echo "  kubectl -n asr get scaledobject,hpa,pods"
echo "  bash scripts/kind_test_autoscale.sh"
