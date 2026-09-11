# The knobs that were previously edited by hand across the YAML in infra/. Codified
# so a change is one reviewed value, not a hunt through four manifests.

variable "namespace" {
  type    = string
  default = "asr"
}

variable "image" {
  description = "Immutable image digest; CI (M7) substitutes this in the manifest path too."
  type        = string
  default     = "ghcr.io/OWNER/asr-serving:REPLACED_BY_CI"
}

variable "gpu_per_pod" {
  type    = number
  default = 1
}

variable "max_num_seqs" {
  description = "vLLM batch slots per pod. Must match the KEDA headroom math (target ~= 0.7 * slots)."
  type        = number
  default     = 32
}

# ---- autoscaling (KEDA ScaledObject) ----
variable "min_replicas" {
  description = "Floor. Never zero: a cold GPU pod is minutes from serving."
  type        = number
  default     = 2
}

variable "max_replicas" {
  type    = number
  default = 16
}

variable "polling_interval_seconds" {
  type    = number
  default = 5
}

variable "cooldown_seconds" {
  type    = number
  default = 300
}

variable "inflight_headroom_threshold" {
  description = "Hold steady-state in-flight requests per fleet at ~70% of slot capacity."
  type        = string
  default     = "22"
}

variable "queue_depth_threshold" {
  description = "Surge trigger: one more pod per N requests waiting. Set low; it is an early warning."
  type        = string
  default     = "4"
}

variable "gpu_guard_threshold" {
  description = "Scale-DOWN guard only (GPU util saturates, so it is never a scale-up signal)."
  type        = string
  default     = "6"
}

variable "prometheus_server" {
  type    = string
  default = "http://prometheus-operated.monitoring.svc:9090"
}
