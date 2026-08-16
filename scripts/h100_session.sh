#!/usr/bin/env bash
# One H100 session, fully scripted, to test whether moving mel extraction to the GPU
# unblocks the throughput ceiling. Everything is staged so the rented box only measures.
#
#   bash scripts/h100_session.sh 2>&1 | tee session.log
#
# THE HYPOTHESIS (ADR-004): the engine is CPU-bound on log-mel extraction, not GPU-bound.
# Measured: 5.93 req/s with GPU at 5% and 23 CPU cores pegged. If that 5% is real headroom,
# 5.93/0.05 = 118 req/s, and the encoder arithmetic agrees: 2.26 TFLOP/clip x 118 req/s =
# 266 TFLOP/s, which is 30-45% of an H100's realistic sustained throughput.
#
# PASS  = arm B beats arm A by >3x with WER unchanged
# WEAK  = arm B beats arm A by 1.3-3x  (bottleneck moved but something else binds)
# FAIL  = arm B within 30% of arm A    (mel was not the bottleneck; look elsewhere)
#
# A WER change of ANY size invalidates the arm regardless of speed. Changing how audio
# becomes features can change transcripts, and a faster wrong answer is not a result.

set -u
export HF_HOME=/workspace/hf
cd /workspace/asr || exit 1
M=openai/whisper-large-v3-turbo
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
mkdir -p results

serve() {   # serve <logfile> <extra args...>
  local log=$1; shift
  pkill -f "vllm-omni serve" 2>/dev/null; sleep 8
  nohup vllm-omni serve "$M" --port 8000 --max-num-seqs 256 \
    --gpu-memory-utilization 0.92 --max-num-batched-tokens 32768 "$@" \
    > "$log" 2>&1 &
  for i in $(seq 1 90); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "  ready after ${i}0s"; return 0; }
    sleep 10
  done
  echo "  SERVER FAILED"; tail -25 "$log"; return 1
}

measure() {  # measure <label> <serverlog>
  local label=$1 slog=$2
  # Sample GPU and the engine's CPU burn together. Either alone is ambiguous; the pair
  # tells you which resource is actually binding.
  nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader -l 2 \
    > "/tmp/gpu_$label.txt" 2>&1 &
  local mon=$!
  ( for i in $(seq 1 60); do
      ps -eo pcpu,comm --sort=-pcpu | grep -m1 -i 'VLLM\|EngineCore' || true
      sleep 2
    done > "/tmp/cpu_$label.txt" 2>&1 ) &
  local cmon=$!

  # 8 client processes: at the throughput we are chasing, one Python client is itself a
  # bottleneck. 8 x c=32 = 256 concurrent, matching max-num-seqs.
  local start=$(date +%s)
  for n in 1 2 3 4 5 6 7 8; do
    python3 bench/loadgen.py --url http://localhost:8000 --model "$M" \
      --audio-dir golden/audio --profile sweep --concurrency-list 32 --per-level 80 \
      --out "results/${label}_c${n}.json" --note "$label client $n" \
      > "/tmp/${label}_c${n}.log" 2>&1 &
  done
  wait
  local end=$(date +%s)
  kill $mon $cmon 2>/dev/null

  local dur=$((end-start))
  echo "  --- $label ---"
  awk -v d="$dur" 'BEGIN{printf "  aggregate: 640 requests in %ds = %.2f req/s\n", d, 640/(d>0?d:1)}'
  echo "  gpu util top3: $(sort -t, -k1 -rn /tmp/gpu_$label.txt 2>/dev/null | head -3 | tr '\n' ' ')"
  echo "  engine cpu top3: $(sort -rn /tmp/cpu_$label.txt 2>/dev/null | head -3 | tr '\n' ' ')"
  grep -h concurrency "/tmp/${label}_c1.log" 2>/dev/null | head -1
}

echo "==========================================================="
echo "ARM A — baseline: mel on CPU (reproduce the 5.93 req/s wall)"
echo "==========================================================="
rm -f "$SP/sitecustomize.py"
serve server_A.log || exit 1
measure armA server_A.log
echo "  WER check:"
python3 ci/wer_gate.py --url http://localhost:8000 --model "$M" \
  --baseline results/m2_baseline_h100.json --tolerance 0.02 \
  --out results/armA_wer.json 2>&1 | grep -E "measured|delta|PASS|FAIL"

echo ""
echo "==========================================================="
echo "ARM B — mel on GPU"
echo "==========================================================="
cp scripts/gpu_mel_patch.py "$SP/sitecustomize.py"
echo "  verifying the patch actually loads:"
python3 -c "
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor as W
print('  patched:', getattr(W, '_asr_gpu_mel_patched', False))
" 2>&1 | tail -2

if ! serve server_B.log; then
  echo "  arm B server failed with GPU mel — retrying with ASR_MEL_RETURN_CPU=1"
  export ASR_MEL_RETURN_CPU=1
  serve server_B.log || { echo "  ARM B UNSERVABLE"; exit 1; }
fi
measure armB server_B.log
echo "  WER check (must be identical to arm A, or the patch is wrong):"
python3 ci/wer_gate.py --url http://localhost:8000 --model "$M" \
  --baseline results/m2_baseline_h100.json --tolerance 0.02 \
  --out results/armB_wer.json 2>&1 | grep -E "measured|delta|PASS|FAIL"

echo ""
echo "==========================================================="
echo "ARM C — arm B plus a concurrency sweep, only if B beat A"
echo "==========================================================="
echo "  (run manually: bench/loadgen.py --profile sweep --concurrency-list 64,128,256,512)"
echo ""
echo "=== SESSION_DONE ==="
