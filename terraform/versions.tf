terraform {
  required_version = ">= 1.5"

  required_providers {
    # gavinbunney/kubectl applies raw manifests (including CRDs like KEDA's
    # ScaledObject and Prometheus's ServiceMonitor) and, unlike the hashicorp
    # kubernetes_manifest resource, computes its plan for *new* objects client-side
    # -- so `terraform plan` works without a live cluster or the CRDs installed.
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
  }
}

provider "kubectl" {
  # No cluster is contacted at plan time. For `terraform apply`, point this at the
  # target cluster (e.g. set load_config_file = true, or KUBE_CONFIG_PATH), or run
  # inside a CI job that already has a kubeconfig.
  load_config_file = false
  apply_retry_count = 3
}
