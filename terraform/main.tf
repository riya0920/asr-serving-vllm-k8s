# Terraform for the Whisper/vLLM-Omni ASR serving stack.
#
# This codifies exactly what infra/ applies by hand: the Kubernetes workload
# (Namespace, Deployment, Service, PodDisruptionBudget), the autoscaling policy
# (KEDA ScaledObject + TriggerAuthentication), and the monitoring hook
# (Prometheus ServiceMonitor). The YAML in infra/ remains the reference; this is
# the same objects as reviewable, parameterized Terraform. It is kept alongside
# the manifests, not in place of Argo CD -- see the README.
#
# `terraform plan` runs with no cluster (see terraform/versions.tf).

locals {
  labels = { app = "whisper-asr" }
}

resource "kubectl_manifest" "namespace" {
  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "Namespace"
    metadata   = { name = var.namespace }
  })
}

resource "kubectl_manifest" "service" {
  depends_on = [kubectl_manifest.namespace]
  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "Service"
    metadata = {
      name = "whisper-asr", namespace = var.namespace, labels = local.labels
    }
    spec = {
      selector = local.labels
      # named port: the ServiceMonitor and the Rollout analysis select on it
      ports = [{ name = "http", port = 8000, targetPort = "http" }]
    }
  })
}

resource "kubectl_manifest" "deployment" {
  depends_on = [kubectl_manifest.namespace]
  yaml_body = yamlencode({
    apiVersion = "apps/v1"
    kind       = "Deployment"
    metadata = {
      name = "whisper-asr", namespace = var.namespace, labels = local.labels
    }
    spec = {
      # No replicas: KEDA owns replica count (Argo CD ignores it too). Setting it
      # here makes every sync fight the autoscaler.
      selector = { matchLabels = local.labels }
      strategy = {
        rollingUpdate = { maxSurge = 1, maxUnavailable = 0 } # GPU pods are slow to replace
      }
      template = {
        metadata = { labels = local.labels }
        spec = {
          terminationGracePeriodSeconds = 120 # let in-flight transcriptions finish
          containers = [{
            name  = "vllm-omni"
            image = var.image
            args = [
              "--model=openai/whisper-large-v3",
              "--task=transcription",
              "--port=8000",
              "--max-num-seqs=${var.max_num_seqs}",
              "--gpu-memory-utilization=0.90",
            ]
            ports     = [{ name = "http", containerPort = 8000 }]
            resources = { limits = { "nvidia.com/gpu" = var.gpu_per_pod } }
            # Whisper-Large takes minutes to load into VRAM: a generous startupProbe
            # keeps liveness from killing the pod mid-load.
            startupProbe   = { httpGet = { path = "/health", port = "http" }, periodSeconds = 10, failureThreshold = 60 }
            readinessProbe = { httpGet = { path = "/health", port = "http" }, periodSeconds = 5, failureThreshold = 2 }
            livenessProbe  = { httpGet = { path = "/health", port = "http" }, periodSeconds = 20, failureThreshold = 3 }
            volumeMounts   = [{ name = "model-cache", mountPath = "/root/.cache/huggingface" }]
          }]
          volumes = [{
            name     = "model-cache"
            hostPath = { path = "/var/cache/hf", type = "DirectoryOrCreate" }
          }]
        }
      }
    }
  })
}

resource "kubectl_manifest" "servicemonitor" {
  depends_on = [kubectl_manifest.namespace]
  yaml_body = yamlencode({
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name = "whisper-asr", namespace = var.namespace, labels = { release = "prometheus" }
    }
    spec = {
      selector = { matchLabels = local.labels }
      # interval must be <= KEDA pollingInterval or KEDA scales on stale numbers
      endpoints = [{ port = "http", path = "/metrics", interval = "${var.polling_interval_seconds}s" }]
    }
  })
}

resource "kubectl_manifest" "pdb" {
  depends_on = [kubectl_manifest.namespace]
  yaml_body = yamlencode({
    apiVersion = "policy/v1"
    kind       = "PodDisruptionBudget"
    metadata   = { name = "whisper-asr", namespace = var.namespace }
    # matches KEDA minReplicaCount -- node drains must not empty the fleet
    spec = { minAvailable = var.min_replicas, selector = { matchLabels = local.labels } }
  })
}

resource "kubectl_manifest" "trigger_auth" {
  depends_on = [kubectl_manifest.namespace]
  yaml_body = yamlencode({
    apiVersion = "keda.sh/v1alpha1"
    kind       = "TriggerAuthentication"
    metadata   = { name = "prometheus-auth", namespace = var.namespace }
    spec       = { secretTargetRef = [] }
  })
}

resource "kubectl_manifest" "scaledobject" {
  depends_on = [kubectl_manifest.deployment, kubectl_manifest.trigger_auth]
  yaml_body = yamlencode({
    apiVersion = "keda.sh/v1alpha1"
    kind       = "ScaledObject"
    metadata   = { name = "whisper-asr", namespace = var.namespace }
    spec = {
      scaleTargetRef  = { name = "whisper-asr" }
      minReplicaCount = var.min_replicas
      maxReplicaCount = var.max_replicas
      pollingInterval = var.polling_interval_seconds
      cooldownPeriod  = var.cooldown_seconds
      advanced = {
        restoreToOriginalReplicaCount = true
        horizontalPodAutoscalerConfig = {
          behavior = {
            scaleUp = {
              stabilizationWindowSeconds = 0 # react on the next evaluation; spikes are step functions
              selectPolicy               = "Max"
              policies = [
                { type = "Percent", value = 100, periodSeconds = 15 }, # double...
                { type = "Pods", value = 4, periodSeconds = 15 },      # ...or +4, whichever is larger
              ]
            }
            scaleDown = {
              stabilizationWindowSeconds = 600 # sluggish on purpose: flapping re-pays cold starts
              selectPolicy               = "Min"
              policies                   = [{ type = "Pods", value = 1, periodSeconds = 120 }]
            }
          }
        }
      }
      # KEDA takes the MAX replica count any trigger asks for -> three triggers, one policy.
      triggers = [
        {
          # 1. warm headroom -- usually decides steady-state replica count
          type = "prometheus", name = "inflight-headroom"
          metadata = {
            serverAddress       = var.prometheus_server
            query               = "sum(vllm:num_requests_running{namespace=\"${var.namespace}\",service=\"whisper-asr\"})"
            threshold           = var.inflight_headroom_threshold
            activationThreshold = "1"
          }
        },
        {
          # 2. queue depth -- leading, proportional surge signal
          type = "prometheus", name = "queue-depth"
          metadata = {
            serverAddress       = var.prometheus_server
            query               = "sum(vllm:num_requests_waiting{namespace=\"${var.namespace}\",service=\"whisper-asr\"})"
            threshold           = var.queue_depth_threshold
            activationThreshold = "1"
          }
        },
        {
          # 3. gpu-util guard -- scale-DOWN only (it saturates, so never a scale-up signal)
          type = "prometheus", name = "gpu-util-guard"
          metadata = {
            serverAddress       = var.prometheus_server
            query               = "avg(DCGM_FI_DEV_GPU_UTIL{namespace=\"${var.namespace}\"})"
            threshold           = var.gpu_guard_threshold
            activationThreshold = "5"
          }
        },
      ]
    }
  })
}
