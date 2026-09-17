#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
DATA_FILE="${DATA_FILE:-guanning-ai/dapo14k}"
OUTPUT_DIR="${OUTPUT_DIR:-slqp_capture_200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

while true; do
  python "$SCRIPT_DIR/collect_slqp_200.py" \
    --stage generate \
    --model "$MODEL" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --samples 200 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"

  set +e
  python "$SCRIPT_DIR/collect_slqp_200.py" \
    --stage replay \
    --model "$MODEL" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --samples 200 \
    --max-new-tokens "$MAX_NEW_TOKENS"
  replay_status=$?
  set -e

  if [[ "$replay_status" -eq 0 ]]; then
    break
  fi
  if [[ "$replay_status" -ne 75 ]]; then
    exit "$replay_status"
  fi
  echo "A degenerate response was rejected; generating a replacement and resuming replay."
done

python "$SCRIPT_DIR/collect_slqp_200.py" \
  --stage package \
  --model "$MODEL" \
  --data "$DATA_FILE" \
  --output "$OUTPUT_DIR" \
  --samples 200 \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo "Collection complete: $OUTPUT_DIR/download.zip"
