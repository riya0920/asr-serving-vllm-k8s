#!/usr/bin/env bash
# Multiple vLLM engines on ONE GPU — attacking the bottleneck ADR-005 actually found.
#
#   bash scripts/multi_instance_session.sh 2>&1 | tee multi.log
#
# WHY THIS AND NOT MORE TUNING
# ADR-005 rejected the CPU hypothesis: moving mel extraction to the GPU cut engine CPU 30%
# and bought 6% throughput. After that patch, on an H100 NVL:
#
#     16 of 192 CPU cores busy      -> CPU is not the limit
#     GPU 50% util at only ~130 W   -> GPU is not the limit either
#
# High utilization at low power means many small kernels with gaps between them. That is a
# SERIALIZED PER-REQUEST PATH inside one engine process, not a resource ceiling. You cannot
# tune your way out of it from the outside — but you can run more than one of it.
#
# Whisper-turbo is 1.6 GB of weights against 94 GB of VRAM, and KV cache peaked at 5%. A
# dozen instances fit comfortably. Each gets its own process, CUDA context and serialized
# path, and they proceed concurrently.
#
# HONESTY NOTE
# This measures aggregate throughput of ONE physical GPU. That is exactly what "requests per
# second per GPU" means, and running multiple replicas per card is a normal production
# pattern. What it is NOT is a single-engine number — say "N engines on one H100" whenever
# quoting it, because a reader will otherwise assume one server process.

set -u
export HF_HOME=/workspace/hf
cd /workspace/asr || exit 1
M=openai/whisper-large-v3-turbo
N=${N:-8}                      # instances on the single GPU
MEM=${MEM:-0.09}               # VRAM fraction each; N * MEM must stay under ~0.92
SEQS=${SEQS:-32}               # batch slots per instance
PERC=${PERC:-40}               # requests per client per instance
mkdir -p results

echo "=== launching $N engines on one GPU (mem $MEM each, $SEQS slots each) ==="
pkill -f "vllm-omni serve" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 8
for i in $(seq 1 "$N"); do
  port=$((8000 + i))
  nohup vllm-omni serve "$M" --port "$port" --max-num-seqs "$SEQS" \
    --gpu-memory-utilization "$MEM" --max-num-batched-tokens 8192 \
    > "server_i${i}.log" 2>&1 &
  echo "  instance $i -> :$port"
done

echo "=== waiting for all $N to report healthy ==="
ready=0
for attempt in $(seq 1 120); do
  ready=0
  for i in $(seq 1 "$N"); do
    curl -sf "localhost:$((8000 + i))/health" >/dev/null 2>&1 && ready=$((ready + 1))
  done
  echo "  ${ready}/${N} ready (t=$((attempt * 10))s)"
  [ "$ready" -eq "$N" ] && break
  sleep 10
done
if [ "$ready" -ne "$N" ]; then
  echo "  only ${ready}/${N} came up — check VRAM: N * MEM must stay under 0.92"
  grep -ihm2 'error\|out of memory' server_i*.log | head -5
  [ "$ready" -eq 0 ] && exit 1
  echo "  continuing with ${ready} instances; scale the result by what actually served"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "=== driving all instances at once ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader -l 2 > /tmp/gpu_multi.txt 2>&1 &
MON=$!
( for i in $(seq 1 90); do ps -eo pcpu,comm --sort=-pcpu | grep -c 'VLLM'; ps -eo pcpu --sort=-pcpu | head -20 | awk '{s+=$1} END {print "TOTALCPU", s}'; sleep 2; done > /tmp/cpu_multi.txt 2>&1 ) &
CMON=$!

pids=""
START=$(date +%s)
for i in $(seq 1 "$ready"); do
  port=$((8000 + i))
  # two client processes per instance so the generator is never the limit
  for c in 1 2; do
    python3 bench/loadgen.py --url "http://localhost:$port" --model "$M" \
      --audio-dir golden/audio --profile sweep --concurrency-list "$SEQS" \
      --per-level "$PERC" --out "results/multi_i${i}_c${c}.json" --note "inst $i client $c" \
      > "/tmp/multi_i${i}_c${c}.log" 2>&1 &
    pids="$pids $!"
  done
done
wait $pids                     # ONLY clients; monitors loop by design (ADR-005 lesson)
END=$(date +%s)
kill $MON $CMON 2>/dev/null

echo ""
echo "=== RESULT ==="
python3 - "$START" "$END" <<'PY'
import glob, json, sys
start, end = int(sys.argv[1]), int(sys.argv[2])
files = sorted(glob.glob("results/multi_i*_c*.json"))
phases = [json.load(open(f))["phases"][0] for f in files]
reqs = sum(p["requests"] for p in phases)
summed = sum(p["achieved_rps"] for p in phases)
wall = reqs / max(1, end - start)
p50 = sorted(p["p50_ms"] for p in phases)
print(f"  clients            {len(phases)}")
print(f"  requests completed {reqs}")
print(f"  wall clock         {end - start}s")
print(f"  AGGREGATE (wall)   {wall:.2f} req/s   <- count what finished, never what was asked")
print(f"  AGGREGATE (summed) {summed:.2f} req/s")
print(f"  median p50         {p50[len(p50)//2]:.0f} ms")
print(f"  errors             {sum(p['errors'] for p in phases)}")
PY
echo "  gpu util top3: $(sort -t, -k1 -rn /tmp/gpu_multi.txt 2>/dev/null | head -3 | tr '\n' ' ')"
echo "  total cpu top3: $(grep TOTALCPU /tmp/cpu_multi.txt 2>/dev/null | sort -k2 -rn | head -3 | tr '\n' ' ')"

echo ""
echo "=== WER on instance 1 (correctness must survive) ==="
python3 ci/wer_gate.py --url http://localhost:8001 --model "$M" \
  --baseline results/m2_baseline_sequential.json --tolerance 0.02 \
  --out results/multi_wer.json 2>&1 | grep -E "measured|delta|PASS|FAIL"

echo "=== MULTI_DONE ==="
