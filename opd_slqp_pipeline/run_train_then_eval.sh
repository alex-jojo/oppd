#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VERL_DIR="$ROOT_DIR/verl"
CALIBRATION_DIR="$ROOT_DIR/slqp_calibration"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
EVAL_GPUS="${EVAL_GPUS:-0}"
N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# Training performance / GPU-memory knobs.  Keep them here so both training
# stages use the same settings and the effective configuration is easy to see.
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-false}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"

N_SAMPLES="${N_SAMPLES:-8}"
MAX_EVAL_PROMPT_TOKENS="${MAX_EVAL_PROMPT_TOKENS:-1024}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
RUN_ROOT="${RUN_ROOT:-/workspace/opd_slqp_run}"
DAPO14K_FILE="${DAPO14K_FILE:-$ROOT_DIR/data/dapo14k_train.parquet}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints" "$RUN_ROOT/merged"

test -f "$CALIBRATION_DIR/manifest.json"
test -f "$CALIBRATION_DIR/scored_responses.jsonl"
python "$ROOT_DIR/prepare_dapo14k.py" --output "$DAPO14K_FILE"
test -f "$DAPO14K_FILE"

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python "$CALIBRATION_DIR/fit_slqp_from_manifest.py" \
  --manifest "$CALIBRATION_DIR/manifest.json" \
  --scores "$CALIBRATION_DIR/scored_responses.jsonl" \
  --output "$RUN_ROOT/qwen3_0.6b_base_slqp_v2.json" \
  --report "$RUN_ROOT/calibration_report.json" \
  --bootstrap-runs 500

cd "$VERL_DIR"

echo "=== Stage 1/4: train vanilla OPD completely ==="
N_GPUS="$N_GPUS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
ENABLE_GRADIENT_CHECKPOINTING="$ENABLE_GRADIENT_CHECKPOINTING" \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
STUDENT_MODEL="$STUDENT_MODEL" \
TEACHER_MODEL="$TEACHER_MODEL" \
TRAIN_FILE="$DAPO14K_FILE" \
VAL_FILES="['$DAPO14K_FILE']" \
OUTPUT_DIR="$RUN_ROOT/checkpoints/vanilla_opd" \
bash examples/g_opd/run_qwen3-0.6b-base-vanilla-opd.sh \
  2>&1 | tee "$RUN_ROOT/logs/vanilla_opd_train.log"

test -f "$RUN_ROOT/checkpoints/vanilla_opd/latest_checkpointed_iteration.txt"

echo "=== Stage 2/4: train SLQP completely ==="
N_GPUS="$N_GPUS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
ENABLE_GRADIENT_CHECKPOINTING="$ENABLE_GRADIENT_CHECKPOINTING" \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
STUDENT_MODEL="$STUDENT_MODEL" \
TRAIN_FILE="$DAPO14K_FILE" \
VAL_FILES="['$DAPO14K_FILE']" \
SLQP_CALIBRATION="$RUN_ROOT/qwen3_0.6b_base_slqp_v2.json" \
OUTPUT_DIR="$RUN_ROOT/checkpoints/slqp" \
bash examples/g_opd/run_qwen3-0.6b-base-slqp.sh \
  2>&1 | tee "$RUN_ROOT/logs/slqp_train.log"

test -f "$RUN_ROOT/checkpoints/slqp/latest_checkpointed_iteration.txt"

echo "=== Stage 3/4: merge both final checkpoints ==="
OPD_STEP="$(tr -d '[:space:]' < "$RUN_ROOT/checkpoints/vanilla_opd/latest_checkpointed_iteration.txt")"
SLQP_STEP="$(tr -d '[:space:]' < "$RUN_ROOT/checkpoints/slqp/latest_checkpointed_iteration.txt")"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$RUN_ROOT/checkpoints/vanilla_opd/global_step_${OPD_STEP}/actor" \
  --target_dir "$RUN_ROOT/merged/vanilla_opd"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$RUN_ROOT/checkpoints/slqp/global_step_${SLQP_STEP}/actor" \
  --target_dir "$RUN_ROOT/merged/slqp"

echo "=== Stage 4/4: evaluate both models with identical settings ==="
cd "$ROOT_DIR"
OPD_MODEL="$RUN_ROOT/merged/vanilla_opd" \
SLQP_MODEL="$RUN_ROOT/merged/slqp" \
EVAL_GPUS="$EVAL_GPUS" \
N_SAMPLES="$N_SAMPLES" \
MAX_PROMPT_TOKENS="$MAX_EVAL_PROMPT_TOKENS" \
MAX_TOKENS="$MAX_EVAL_TOKENS" \
MAX_NUM_SEQS="$MAX_NUM_SEQS" \
TEMPERATURE=1.0 \
TOP_P=1.0 \
SEED=42 \
OUTPUT_ROOT="$RUN_ROOT/eval_outputs/opd_vs_slqp" \
bash math_eval/run_eval_opd_slqp.sh \
  2>&1 | tee "$RUN_ROOT/logs/opd_slqp_eval.log"

cd "$RUN_ROOT"
python3 -m zipfile -c \
  "$RUN_ROOT/opd_slqp_eval_results.zip" \
  eval_outputs \
  calibration_report.json \
  qwen3_0.6b_base_slqp_v2.json \
  logs

echo "COMPLETE"
echo "Download: $RUN_ROOT/opd_slqp_eval_results.zip"
