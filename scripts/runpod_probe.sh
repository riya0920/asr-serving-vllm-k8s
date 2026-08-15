#!/usr/bin/env bash
# M1 substrate probe — can a real Kubernetes node exist on this box?
#
# Run this on the CHEAPEST GPU pod the provider will rent you. It costs a few cents
# and decides the entire infrastructure plan (M5-M8).
#
#   bash scripts/runpod_probe.sh 2>&1 | tee results/m1_substrate_probe.txt
#
# Verdict at the bottom:
#   PASS  -> build the Kubernetes half here
#   FAIL  -> model work here, cluster work on bare metal or a root-access VM

set -u

PASS=0
FAIL=0
WARN=0

hdr()  { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  [PASS] %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  [WARN] %s\n' "$1"; WARN=$((WARN+1)); }

hdr "Host identity"
printf '  kernel : %s\n' "$(uname -r)"
printf '  distro : %s\n' "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
printf '  cpus   : %s\n' "$(nproc 2>/dev/null || echo '?')"
printf '  memory : %s\n' "$(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
if [ -f /.dockerenv ] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
  warn "running inside a container — this is the thing that usually blocks k3s"
else
  ok "not obviously containerized"
fi

hdr "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  ok "nvidia-smi works"
else
  bad "no nvidia-smi — wrong image or no GPU attached"
fi

hdr "Privileges (k3s needs these)"
if command -v capsh >/dev/null 2>&1; then
  CAPS=$(capsh --print 2>/dev/null | grep -m1 'Current:')
  printf '  %s\n' "$CAPS"
  case "$CAPS" in
    *cap_sys_admin*) ok "CAP_SYS_ADMIN present" ;;
    *)               bad "CAP_SYS_ADMIN missing — kubelet cannot manage mounts/cgroups" ;;
  esac
else
  warn "capsh not installed (apt-get install -y libcap2-bin) — checking indirectly"
  if mount -t tmpfs tmpfs /mnt 2>/dev/null; then umount /mnt; ok "can mount — privileged enough"
  else bad "cannot mount tmpfs — not privileged"; fi
fi

if [ "$(id -u)" = "0" ]; then ok "running as root"; else bad "not root"; fi

hdr "cgroups"
if [ -d /sys/fs/cgroup ]; then
  if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    printf '  version: v2, controllers: %s\n' "$(cat /sys/fs/cgroup/cgroup.controllers)"
  else
    printf '  version: v1\n'
  fi
  if mkdir -p /sys/fs/cgroup/probe-test 2>/dev/null; then
    rmdir /sys/fs/cgroup/probe-test 2>/dev/null
    ok "cgroup hierarchy is writable"
  else
    bad "cgroup hierarchy is read-only — kubelet will refuse to start"
  fi
else
  bad "/sys/fs/cgroup missing entirely"
fi

hdr "Kernel modules and networking"
for m in br_netfilter overlay; do
  if lsmod 2>/dev/null | grep -q "^$m" || modprobe "$m" 2>/dev/null; then
    ok "module $m available"
  else
    warn "module $m not loadable — k3s may fall back or fail on pod networking"
  fi
done
[ -w /proc/sys/net/ipv4/ip_forward ] && ok "can set ip_forward" || bad "cannot set ip_forward"

hdr "Installing k3s (the actual test)"
if curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --disable=servicelb" sh - >/tmp/k3s-install.log 2>&1; then
  ok "k3s installer completed"
  printf '  waiting up to 120s for node Ready...\n'
  READY=""
  for _ in $(seq 1 24); do
    if k3s kubectl get nodes 2>/dev/null | grep -q ' Ready '; then READY=1; break; fi
    sleep 5
  done
  if [ -n "$READY" ]; then
    ok "node reports Ready"
    k3s kubectl get nodes -o wide 2>/dev/null | sed 's/^/  /'
  else
    bad "node never became Ready"
    printf '  --- last 30 lines of k3s log ---\n'
    journalctl -u k3s --no-pager -n 30 2>/dev/null | sed 's/^/  /' || tail -n 30 /tmp/k3s-install.log | sed 's/^/  /'
  fi
else
  bad "k3s installer failed"
  tail -n 30 /tmp/k3s-install.log | sed 's/^/  /'
fi

hdr "NVIDIA device plugin (does k8s actually see the GPU?)"
if k3s kubectl get nodes >/dev/null 2>&1; then
  k3s kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/deployments/static/nvidia-device-plugin.yml >/dev/null 2>&1
  printf '  waiting up to 90s for nvidia.com/gpu to appear on the node...\n'
  GPUS=""
  for _ in $(seq 1 18); do
    GPUS=$(k3s kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
    [ -n "$GPUS" ] && [ "$GPUS" != "0" ] && break
    sleep 5
  done
  if [ -n "$GPUS" ] && [ "$GPUS" != "0" ]; then
    ok "node advertises nvidia.com/gpu: $GPUS"
  else
    bad "device plugin never advertised a GPU (needs nvidia-container-runtime as the container runtime)"
    k3s kubectl -n kube-system logs -l name=nvidia-device-plugin-ds --tail=20 2>/dev/null | sed 's/^/  /'
  fi
fi

hdr "VERDICT"
printf '  pass=%s  fail=%s  warn=%s\n\n' "$PASS" "$FAIL" "$WARN"
if [ "$FAIL" -eq 0 ]; then
  printf '  PASS — build the Kubernetes half here (M5-M8).\n'
  printf '  Next: rent the multi-GPU box, run this again to confirm, then M5.\n'
else
  printf '  FAIL — this substrate cannot host a real Kubernetes node.\n'
  printf '  Keep model work (M2-M4) here; move cluster work to bare metal or a\n'
  printf '  root-access VM. Record the choice in docs/adr-001-substrate.md.\n'
fi

hdr "Cleanup"
printf '  to remove k3s:  /usr/local/bin/k3s-uninstall.sh\n'
printf '  STOP THE POD when done — it bills while idle.\n'
