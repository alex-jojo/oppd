#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="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"
export WANDB_MODE="online"
export WANDB_PROJECT="icml_long"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B-Base}"
SLQP_CALIBRATION="${SLQP_CALIBRATION:?Set SLQP_CALIBRATION to the frozen JSON for this student checkpoint}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the prepared guanning-ai/dapo14k parquet}"
VAL_FILES="${VAL_FILES:-['$TRAIN_FILE']}"
N_GPUS="${N_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-/G-OPD-checkpoints/Qwen3-0.6B-Base-SLQP}"
SAVE_FREQ="${SAVE_FREQ:-50}"
MAX_ACTOR_CKPTS="${MAX_ACTOR_CKPTS:-2}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=true \
    data.seed=42 \
    data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=false \
    actor_rollout_ref.actor.policy_loss.slqp.enabled=true \
    actor_rollout_ref.actor.policy_loss.slqp.calibration_path="$SLQP_CALIBRATION" \
    actor_rollout_ref.actor.policy_loss.slqp.loss_weight=1.0 \
    actor_rollout_ref.actor.policy_loss.slqp.slqp_only=true \
    actor_rollout_ref.actor.policy_loss.slqp.huber_delta=1.0 \
    actor_rollout_ref.actor.policy_loss.slqp.min_response_tokens=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=icml_long \
    trainer.experiment_name=qwen3-0.6b-base-slqp-only \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPTS" \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "$@"
