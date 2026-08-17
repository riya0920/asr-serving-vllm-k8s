#!/usr/bin/env bash
# ENGINE A/B: is 118 req/s an engine problem or a hardware problem?
#
#   bash scripts/engine_ab.sh 2>&1 | tee engine_ab.log
#
# One variable. Same GPU, same 50 clips, same load generator, same WER gate. Only the
# serving engine changes.
#
#   ARM A  vLLM-Omni        whisper-large-v3-turbo   (13.07 req/s on A40, our measured ceiling)
#   ARM B  faster-whisper   large-v3-turbo (CT2)     batched inference pipeline
#
# WHAT THE OUTCOMES MEAN
#   B >= 3x A   the ceiling was vLLM, not the GPU, and the throughput target is a question of
#               picking the right engine: (B on this card) x (H100/this-card ratio).
#   B ~ A       the ceiling is the model on this hardware, and no engine swap reaches the
#               target for Whisper-Large at 30s clips.
#   B < A       vLLM was already the better choice; keep it and keep the honest number.
#
# WER is checked on both arms. A faster engine that transcribes worse has not won anything,
# and CTranslate2 quantizes differently, so this is a real risk rather than a formality.

set -u
export HF_HOME=${HF_HOME:-/workspace/hf}
cd "$(dirname "$0")/.." || exit 1
GOLDEN=${GOLDEN:-golden}
RESULTS=${RESULTS:-results}
mkdir -p "$RESULTS"

wait_health() {  # wait_health <port> <label>
  for i in $(seq 1 90); do
    curl -sf "localhost:$1/health" >/dev/null 2>&1 && { echo "  $2 ready after ${i}0s"; return 0; }
    sleep 10
  done
  echo "  $2 FAILED to start"; return 1
}

measure() {  # measure <port> <label> <model-name-for-api>
  local port=$1 label=$2 model=$3
  nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader -l 2 \
    > "/tmp/gpu_$label.txt" 2>&1 &
  local mon=$!
  local pids="" start=$(date +%s)
  # 4 client processes: one Python client cannot saturate a fast engine, and a
  # client-bound number would answer the wrong question entirely.
  for n in 1 2 3 4; do
    python3 bench/loadgen.py --url "http://localhost:$port" --model "$model" \
      --audio-dir "$GOLDEN/audio" --profile sweep --concurrency-list 32 --per-level 60 \
      --out "$RESULTS/${label}_c$n.json" --note "$label client $n" \
      > "/tmp/${label}_c$n.log" 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  local end=$(date +%s)
  kill $mon 2>/dev/null
  python3 - "$label" "$((end-start))" <<'PY'
import glob, json, sys
label, dur = sys.argv[1], int(sys.argv[2])
ph = [json.load(open(f))["phases"][0] for f in sorted(glob.glob(f"results/{label}_c*.json"))]
reqs = sum(p["requests"] for p in ph)
print(f"  {label}: {reqs} requests in {dur}s = {reqs/max(1,dur):.2f} req/s aggregate")
print(f"  {label}: median p50 {sorted(p['p50_ms'] for p in ph)[len(ph)//2]:.0f} ms, "
      f"errors {sum(p['errors'] for p in ph)}")
PY
  echo "  gpu top3: $(sort -t, -k1 -rn /tmp/gpu_$label.txt 2>/dev/null | head -3 | tr '\n' ' ')"
}

echo "==================================================================="
echo "ARM A — vLLM-Omni"
echo "==================================================================="
pkill -9 -f "vllm-omni serve" 2>/dev/null; pkill -9 -f fasterwhisper_server 2>/dev/null; sleep 8
nohup vllm-omni serve openai/whisper-large-v3-turbo --port 8000 \
  --max-num-seqs 256 --gpu-memory-utilization 0.85 --max-num-batched-tokens 32768 \
  > /tmp/armA_server.log 2>&1 &
if wait_health 8000 "vLLM"; then
  measure 8000 engineA openai/whisper-large-v3-turbo
  echo "  WER:"
  python3 ci/wer_gate.py --url http://localhost:8000 --model openai/whisper-large-v3-turbo \
    --golden "$GOLDEN" --baseline "$RESULTS/m2_baseline_sequential.json" --tolerance 0.02 \
    --out "$RESULTS/engineA_wer.json" 2>&1 | grep -E "measured|delta|PASS|FAIL"
fi
pkill -9 -f "vllm-omni serve" 2>/dev/null; sleep 10

echo ""
echo "==================================================================="
echo "ARM B — faster-whisper (CTranslate2)"
echo "==================================================================="
pip install --break-system-packages -q faster-whisper 2>&1 | tail -1
nohup python3 bench/fasterwhisper_server.py --port 8001 --model large-v3-turbo \
  --batch-size 32 > /tmp/armB_server.log 2>&1 &
if wait_health 8001 "faster-whisper"; then
  measure 8001 engineB large-v3-turbo
  echo "  WER:"
  python3 ci/wer_gate.py --url http://localhost:8001 --model large-v3-turbo \
    --golden "$GOLDEN" --baseline "$RESULTS/m2_baseline_sequential.json" --tolerance 0.02 \
    --out "$RESULTS/engineB_wer.json" 2>&1 | grep -E "measured|delta|PASS|FAIL"
else
  echo "  arm B failed to start:"; tail -15 /tmp/armB_server.log
fi

echo ""
echo "==================================================================="
echo "VERDICT"
echo "==================================================================="
python3 - <<'PY'
import glob, json
def agg(label):
    fs = sorted(glob.glob(f"results/{label}_c*.json"))
    if not fs: return None
    ph = [json.load(open(f))["phases"][0] for f in fs]
    return sum(p["achieved_rps"] for p in ph)
a, b = agg("engineA"), agg("engineB")
if a and b:
    print(f"  vLLM-Omni      {a:.2f} req/s")
    print(f"  faster-whisper {b:.2f} req/s")
    print(f"  ratio          {b/a:.2f}x")
    print()
    if b >= 3*a:
        print("  >>> ENGINE-BOUND. The GPU was never the ceiling; vLLM was.")
        print(f"  >>> extrapolating to H100 (~2.5-3.5x): {b*2.5:.0f}-{b*3.5:.0f} req/s")
    elif b >= 1.3*a:
        print("  >>> faster, but not enough. 118 still out of reach on this card.")
    else:
        print("  >>> NOT engine-bound. The ceiling is the model on this hardware.")
    print()
    print("  Whichever wins, the WER lines above decide whether it counts.")
else:
    print("  one arm produced no data — see the logs above")
PY
echo "=== ENGINE_AB_DONE ==="
