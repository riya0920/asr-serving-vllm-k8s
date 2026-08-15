#!/usr/bin/env bash
# Drive the stub fleet with the real 8x step-function profile and record what the autoscaler
# actually did, second by second.
#
# This is a dress rehearsal for M6. It cannot produce a latency number that means anything
# (the stub's service time is made up), but it DOES prove the loop end to end: metric
# exposed -> scraped -> queried by KEDA -> HPA replica count -> new pods -> queue drains.
# Every failure mode in that chain is cheaper to find here.
#
#   bash scripts/kind_test_autoscale.sh
#
# What to look for, in order:
#   1. does queue depth actually rise?     (if not, the load is too light to test anything)
#   2. does the HPA replica count follow?  (if not, the KEDA query is returning nothing)
#   3. how many seconds between the queue rising and new pods being Ready?
#   4. does the fleet hold, not flap, when the spike ends?

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/results/m5_kind_autoscale.csv"
DURATION=${DURATION:-330}
RPS=${RPS:-6}

command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }

echo "==> port-forwarding the service to localhost:8000"
kubectl -n asr port-forward svc/whisper-asr 8000:8000 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
sleep 3

echo "==> sampling cluster state every second -> $OUT"
mkdir -p "$(dirname "$OUT")"
echo "t,desired,current,ready,waiting,running" > "$OUT"
(
  for t in $(seq 0 "$DURATION"); do
    HPA=$(kubectl -n asr get hpa -o jsonpath='{.items[0].status.desiredReplicas} {.items[0].status.currentReplicas}' 2>/dev/null || echo "0 0")
    READY=$(kubectl -n asr get pods -l app=whisper-asr \
      -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' 2>/dev/null \
      | grep -c True || true)
    M=$(curl -s --max-time 2 localhost:8000/debug || echo '{}')
    W=$(echo "$M" | sed -n 's/.*"waiting": *\([0-9]*\).*/\1/p'); W=${W:-0}
    R=$(echo "$M" | sed -n 's/.*"running": *\([0-9]*\).*/\1/p'); R=${R:-0}
    echo "$t,${HPA% *},${HPA#* },$READY,$W,$R" >> "$OUT"
    sleep 1
  done
) &
SAMPLER=$!

echo "==> driving the 8x step-function profile (base ${RPS} rps -> $((RPS*8)) rps)"
python "${ROOT}/bench/loadgen.py" \
  --url http://localhost:8000 \
  --audio-dir "${ROOT}/golden/audio" \
  --profile spike-8x --rps "$RPS" \
  --out "${ROOT}/results/m5_kind_spike.json" \
  --note "kind + stub server; validates the autoscaling loop, NOT latency"

wait $SAMPLER 2>/dev/null || true

echo
echo "==> autoscaler timeline"
python - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit("no samples captured")

print(f"{'t':>4} {'desired':>8} {'ready':>6} {'queue':>6}  {'':<30}")
prev = None
for r in rows:
    t, d, ready, w = int(r['t']), r['desired'], r['ready'], int(r['waiting'] or 0)
    mark = ''
    if prev is not None and d != prev:
        mark = f"  <-- scaled {prev} -> {d}"
    prev = d
    if t % 5 == 0 or mark:
        print(f"{t:>4} {d:>8} {ready:>6} {w:>6}  {'#'*min(30, w)}{mark}")

first_q = next((int(r['t']) for r in rows if int(r['waiting'] or 0) > 0), None)
base = rows[0]['desired']
first_s = next((int(r['t']) for r in rows if r['desired'] != base), None)
peak = max(int(r['desired'] or 0) for r in rows)
print()
print(f"  first queued request     t={first_q}")
print(f"  first scale-up decision  t={first_s}")
if first_q is not None and first_s is not None:
    print(f"  autoscaler reaction      {first_s - first_q}s   <- the number that matters")
print(f"  peak desired replicas    {peak}")
print()
print("  reaction time here is a FLOOR, not the real number: the stub loads in 20s where")
print("  Whisper-Large takes minutes. Scale-up latency on real hardware is measured in M5.")
PY
