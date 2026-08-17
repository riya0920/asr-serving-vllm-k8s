#!/usr/bin/env bash
# Whisper on Kubernetes, on a GPU node — closing the integration gap.
#
#   bash scripts/gpu_k8s_session.sh 2>&1 | tee gpu_k8s.log
#
# THE GAP THIS CLOSES
# The serving and autoscaling halves had been measured on two different machines: Whisper on
# rented GPUs with no Kubernetes, and KEDA autoscaling a stub on a cluster with no GPU. This
# puts both on one host so the whole path is exercised together.
#
# Requires a host where root means root AND an NVIDIA GPU is present — a real VM, not a
# container. Every RunPod product available self-serve fails the first condition (ADR-001);
# WSL2 satisfied it but has no NVIDIA GPU on this laptop.
#
# THE TRICK THAT MAKES ONE GPU ENOUGH
# The NVIDIA device plugin can time-slice a single physical GPU into N allocatable
# `nvidia.com/gpu` resources. KEDA then scales real Whisper pods 2 -> N on one card, so the
# 8x spike test runs against actual model servers instead of a stub. Time-sliced replicas
# contend for the same SMs, so per-pod throughput drops — this proves the autoscaling and
# scheduling path end to end, not that N pods give N times the throughput.

set -u
export DEBIAN_FRONTEND=noninteractive
REPLICAS=${REPLICAS:-8}          # time-sliced virtual GPUs
MODEL=${MODEL:-openai/whisper-large-v3-turbo}

echo "=== 0. preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no NVIDIA GPU visible — wrong host"; exit 1; }
[ "$(id -u)" = "0" ] || { echo "must run as root"; exit 1; }

echo ""
echo "=== 1. k3s ==="
if ! command -v k3s >/dev/null 2>&1; then
  # iptables first: without it k3s starts anyway and pod sandboxes churn forever with
  # SandboxChanged, which presents as an application crash loop (ADR-008).
  apt-get update -qq && apt-get install -y -qq iptables curl
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --disable=servicelb --write-kubeconfig-mode 644" sh -
fi
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
for i in $(seq 1 30); do kubectl get nodes 2>/dev/null | grep -q ' Ready ' && break; sleep 10; done
kubectl get nodes -o wide | tail -2

echo ""
echo "=== 2. NVIDIA container runtime + device plugin with time-slicing ==="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=containerd --config /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null || true
systemctl restart k3s; sleep 25

kubectl apply -f - <<YAML
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: kube-system
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: ${REPLICAS}
YAML
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/deployments/static/nvidia-device-plugin.yml 2>&1 | tail -2
kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"nvidia-device-plugin-ctr","args":["--config-file=/config/any"],"volumeMounts":[{"name":"tsconfig","mountPath":"/config"}]}],"volumes":[{"name":"tsconfig","configMap":{"name":"time-slicing-config"}}]}}}}' 2>/dev/null

echo "  waiting for the node to advertise GPUs..."
for i in $(seq 1 24); do
  G=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
  [ -n "$G" ] && [ "$G" != "0" ] && { echo "  node advertises nvidia.com/gpu: $G"; break; }
  sleep 10
done
[ -z "${G:-}" ] || [ "$G" = "0" ] && { echo "  device plugin never advertised a GPU"; \
  kubectl -n kube-system logs -l name=nvidia-device-plugin-ds --tail=15 2>/dev/null | tail -8; }

echo ""
echo "=== 3. REAL Whisper pods (not the stub) ==="
kubectl create namespace asr --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f - <<YAML
apiVersion: v1
kind: Service
metadata:
  name: whisper-asr
  namespace: asr
  labels: { app: whisper-asr }
spec:
  selector: { app: whisper-asr }
  ports: [{ name: http, port: 8000, targetPort: http }]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whisper-asr
  namespace: asr
  labels: { app: whisper-asr }
spec:
  replicas: 2
  selector:
    matchLabels: { app: whisper-asr }
  template:
    metadata:
      labels: { app: whisper-asr }
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "${MODEL}", "--port", "8000",
                 "--max-num-seqs", "16", "--gpu-memory-utilization", "0.10"]
          ports: [{ name: http, containerPort: 8000 }]
          resources:
            limits: { nvidia.com/gpu: 1 }
          env:
            - { name: HF_HOME, value: /models }
          startupProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 10
            failureThreshold: 90     # weights load takes minutes; liveness alone crash-loops
          readinessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 5
          volumeMounts: [{ name: models, mountPath: /models }]
      volumes:
        - name: models
          hostPath: { path: /var/cache/hf, type: DirectoryOrCreate }
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: whisper-asr
  namespace: asr
  labels: { release: prometheus }
spec:
  selector:
    matchLabels: { app: whisper-asr }
  endpoints: [{ port: http, path: /metrics, interval: 5s }]
YAML

echo "  waiting for real Whisper pods (weights download + load, several minutes)..."
kubectl -n asr rollout status deploy/whisper-asr --timeout=900s 2>&1 | tail -2
kubectl -n asr get pods -o wide | tail -4

echo ""
echo "=== 4. proof it is really transcribing, from inside the cluster ==="
IP=$(kubectl -n asr get pods -l app=whisper-asr -o jsonpath='{.items[0].status.podIP}')
echo "  pod ip: $IP"
curl -s --max-time 120 -X POST "http://$IP:8000/v1/audio/transcriptions" \
  -F "file=@golden/audio/clip_000.wav" -F "model=${MODEL}" | head -c 250
echo ""
echo "  metric check (KEDA depends on this existing):"
curl -s --max-time 10 "http://$IP:8000/metrics" | grep -E '^vllm:num_requests_(waiting|running)' | head -2

echo ""
echo "=== 5. Prometheus + KEDA + the project ScaledObject, unmodified ==="
# (install helm/prometheus/keda exactly as scripts/wsl_cluster.sh does, then:)
echo "  kubectl apply -f infra/keda/scaledobject.yaml"
echo "  bash scripts/kind_test_autoscale.sh   # 8x spike against REAL Whisper pods"
echo ""
echo "=== GPU_K8S_DONE ==="
