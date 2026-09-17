#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPD_MODEL="${OPD_MODEL:?Set OPD_MODEL to the merged Hugging Face OPD directory}"
SLQP_MODEL="${SLQP_MODEL:?Set SLQP_MODEL to the merged Hugging Face SLQP directory}"
EVAL_GPUS="${EVAL_GPUS:-0}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_outputs/opd_vs_slqp}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"

run_one_model() {
  local model_name="$1"
  local model_path="$2"
  local dataset_name
  local input_file

  for dataset_name in aime24 aime25 hmmt25_feb hmmt25_nov; do
    input_file="$DATA_ROOT/$dataset_name/test.jsonl"
    if [[ ! -f "$input_file" ]]; then
      echo "Missing evaluation data: $input_file" >&2
      exit 1
    fi
    mkdir -p "$OUTPUT_ROOT/$dataset_name"
    echo "Evaluating $model_name on $dataset_name"
    CUDA_VISIBLE_DEVICES="$EVAL_GPUS" python3 "$SCRIPT_DIR/eval_math.py" \
      --input_file "$input_file" \
      --model_path "$model_path" \
      --output_file "$OUTPUT_ROOT/$dataset_name/$model_name.jsonl" \
      --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
      --max_tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --max_num_seqs "$MAX_NUM_SEQS" \
      --n "$N_SAMPLES" \
      --begin_idx -1 \
      --end_idx -1 \
      --seed "$SEED" \
      2>&1 | tee "$OUTPUT_ROOT/$dataset_name/$model_name.log"
  done
}

# Sequential on purpose: both models use identical settings without competing
# for GPU memory or changing the effective vLLM cache budget.
run_one_model "vanilla_opd" "$OPD_MODEL"
run_one_model "slqp" "$SLQP_MODEL"

echo "Evaluation complete: $OUTPUT_ROOT"
