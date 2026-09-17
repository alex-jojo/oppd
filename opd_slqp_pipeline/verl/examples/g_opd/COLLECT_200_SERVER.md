# 200 条逐 token hidden 收集：新服务器命令

该收集任务不需要安装 verl，也不使用 W&B。模型和 dapo14k 会自动下载。

## 1. 上传与解压

将 slqp_collect_200_server_20260917.zip 上传到服务器的 /workspace，然后运行：

    cd /workspace
    python3 -m zipfile -e slqp_collect_200_server_20260917.zip slqp_collect_200
    cd /workspace/slqp_collect_200

## 2. 创建环境

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -n slqp-collect python=3.10 -y
    conda activate slqp-collect
    python -m pip install --upgrade pip
    python -m pip install "vllm==0.10.0" "transformers>=4.51,<5" "datasets>=3,<5" "numpy<2" safetensors
    python -m pip check

检查 GPU：

    export CUDA_VISIBLE_DEVICES=0
    nvidia-smi
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 3. 正式收集

    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/workspace/hf_cache
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    bash run_collect_slqp_200.sh 2>&1 | tee collect.log

默认行为：

- 自动下载 Qwen/Qwen3-0.6B-Base。
- 自动下载 guanning-ai/dapo14k。
- vLLM 一次生成 200 条回答。
- 默认每条最多 4096 个 response token。
- vLLM 退出后再启动 Hugging Face replay。
- 保存每个 response token 的最终层 hidden，不使用 stride。
- 轨迹特征和 `trajectory_response_length` 只统计实际生成且非 tokenizer special 的 response token；prompt、padding、EOS/控制 token 不计入。
- 同时保存原始 response mean/last hidden 与排除 special token 后的 trajectory mean/last hidden、trajectory mask。
- 自动生成 download.zip。
- 没有文件大小硬上限。

运行中断后，重新执行同一个 bash 命令即可。已经写入的 response batch 和已经 replay 的单条 hidden 会被复用。

如果某条回答只生成 EOS/控制 token，脚本会把它记录到 `rejected_responses.jsonl`，自动换一道题补生成，并继续复用已经完成的 hidden；最终包仍严格包含 200 条有效轨迹。

## 4. 结果

完成时会打印 archive_bytes，并生成：

    /workspace/slqp_collect_200/slqp_capture_200/download.zip

只需下载这个 ZIP。模型缓存不需要下载。

如果想改生成 batch：

    GENERATION_BATCH_SIZE=16 bash run_collect_slqp_200.sh 2>&1 | tee collect.log

如果想改最大回答长度：

    MAX_NEW_TOKENS=8192 bash run_collect_slqp_200.sh 2>&1 | tee collect.log

不建议在第一次收集时同时改模型、采样参数或数据集，否则会改变校准数据分布。
