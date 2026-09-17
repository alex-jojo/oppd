# OPD to SLQP to unified evaluation: server commands

The upload package contains code, four math evaluation datasets, the 200
GPT-5.5 scores, the 54 KB trajectory-feature manifest, and the calibration
script. It deliberately excludes the 1.3 GB per-token hidden bundle.

## 1. Upload and extract

Upload opd_slqp_full_pipeline_20260917.zip to /workspace, then run:

    cd /workspace

    python3 -m zipfile -e \
      /workspace/opd_slqp_full_pipeline_20260917.zip \
      /workspace/opd_slqp_pipeline

## 2. Create the environment

    source "$(conda info --base)/etc/profile.d/conda.sh"

    conda create -n opd-slqp python=3.10 -y
    conda activate opd-slqp

    cd /workspace/opd_slqp_pipeline/verl

    python -m pip install --upgrade pip

    USE_MEGATRON=0 USE_SGLANG=0 \
    bash scripts/install_vllm_sglang_mcore.sh

    python -m pip install -e . --no-deps
    python -m pip install math-verify
    python -m pip check

The packaged installer pins transformers==4.52.4, tokenizers==0.21.2, and
huggingface-hub==0.33.1. Do not upgrade Transformers to 5.x: this VERL version
still imports AutoModelForVision2Seq.

## 3. Training data

The full runner downloads guanning-ai/dapo14k and converts all of its rows to
the VERL RL parquet schema at data/dapo14k_train.parquet. Both OPD and SLQP use
that exact same parquet for one epoch. DeepMath is not downloaded or used.
The student and teacher model weights are downloaded automatically on first use.

## 4. Run the full single-GPU experiment

The default below is for one 80 GB GPU. It runs in the foreground and stops
immediately if any stage fails.

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate opd-slqp

    cd /workspace/opd_slqp_pipeline

    export TRAIN_GPUS=0
    export EVAL_GPUS=0
    export N_GPUS=1
    export TRAIN_BATCH_SIZE=128
    export MAX_PROMPT_LENGTH=1024
    export MAX_RESPONSE_LENGTH=2048
    export N_SAMPLES=8
    export MAX_EVAL_PROMPT_TOKENS=1024
    export MAX_EVAL_TOKENS=8192
    export MAX_NUM_SEQS=64
    export RUN_ROOT=/workspace/opd_slqp_run

    bash run_train_then_eval.sh

The script runs this order:

1. refit the frozen SLQP calibration parameters (there is no score-threshold gate);
2. finish all vanilla OPD training;
3. finish all pure-SLQP training;
4. merge both final checkpoints;
5. evaluate both with identical sampling on AIME24, AIME25, HMMT25-Feb,
   and HMMT25-Nov;
6. package the generations, metrics, calibration, and logs.

The defaults are fixed as follows: each training prompt produces one rollout;
evaluation produces eight independent responses per problem and reports
mean@8, pass@8, and mathematical-answer majority-vote maj@8. Training uses a
1024-token prompt cap and 2048-token response cap. Evaluation uses the same
1024-token prompt cap and an 8192-token response cap. Both sides use the
tokenizer's chat template with add_generation_prompt=true and
enable_thinking=false.

Both training launchers have trainer.val_before_train=false and
trainer.test_freq=-1. Formal evaluation therefore starts only after both
training jobs have completed.

For long single-GPU runs, both methods save every 500 steps and automatically
keep only the latest two actor checkpoints. The final step is always saved.
This preserves resume capability without filling the disk with checkpoints.

For eight GPUs, change only:

    export TRAIN_GPUS=0,1,2,3,4,5,6,7
    export N_GPUS=8
    export TRAIN_BATCH_SIZE=128

Evaluation remains sequential and uses the GPU list in EVAL_GPUS.

## 5. Final result

After COMPLETE is printed, download:

    /workspace/opd_slqp_run/opd_slqp_eval_results.zip

Do not download checkpoints or model caches unless they are separately needed.

W&B is already configured in both training launchers with project icml_long.
The rule-based math reward remains a monitoring metric and is not used as the
OPD or SLQP training signal.

The single-GPU launchers use micro-batches of four. The 4B OPD teacher stays on
GPU, compatible Qwen3 token IDs are reused without per-step decode/re-tokenize,
and the exact one-epoch/one-minibatch OPD path skips the otherwise unused
student old-log-prob replay and generic GRPO advantage calculation. The
student rollout, teacher token log-probabilities, and student backward update
remain unchanged.
