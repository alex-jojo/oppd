# SLQP 与 vanilla OPD 实现说明

## 已确认的 vanilla OPD

仓库原来的 run_qwen3-4b-g-opd.sh 前半段确实是标准 OPD：单条 rollout、用 teacher 在同一 sampled token 上的 log-prob 构造 reverse-KL advantage，且 lambda_vals=1.0。但它固定使用 Qwen3-1.7B student，并且同一个脚本还会接着运行 G-OPD。

新增的 run_qwen3-0.6b-base-vanilla-opd.sh 是独立的干净基线：

- student：Qwen/Qwen3-0.6B-Base
- teacher：Qwen/Qwen3-4B
- n=1
- only_reverse_kl_advantages=true
- lambda_vals=1.0
- 不启用 SLQP
- 不启用 base-model correction、多 teacher、G-OPD 或 ExOPD

teacher 可通过 TEACHER_MODEL 改成 math checkpoint，但基线默认严格对应 4B 到 0.6B-Base。

## SLQP 的实际训练路径

SLQP 使用 student rollout 的固定 token，在 actor 更新阶段做一次有梯度 replay。代码从同一次 forward 的最终归一化层截取 hidden，不会为了 SLQP 再跑第二次模型。

纯 SLQP 会跳过 verl 默认的 old-log-prob 无梯度 replay，也不会计算随后会被丢弃的 PPO/GRPO advantage、PPO loss、teacher KL、entropy 或完整 response-vocabulary logits。Qwen3 CausalLM 只保留一个末位 logits 以维持标准 forward 接口；真正参与反向传播的是同一次 backbone forward 中截取的最终层 hidden。verl 通用监控需要的 advantages/returns 与不适用的 PPO、KL、teacher 指标使用零占位，但真实数学 reward、SLQP loss、hidden 特征、梯度范数和长度指标不会填零。

对每条回答，prompt 最后一个 token 的 hidden 作为 h0，并且 stop-gradient。回答 token 的 hidden 为 h1...hT。计算四个可微特征：

1. response 到 h0 的归一化距离均值
2. response 到 h0 的归一化距离标准差
3. 相邻 hidden 归一化距离均值
4. 相邻 hidden 归一化距离标准差

使用冻结校准文件中的均值、标准差和 PCA 方向得到标量质量。只在质量低于好簇边界时施加 one-sided Huber；已经处于好侧的回答损失严格为零。没有 q10、q50、q90、窗口切分或 slope。

质量几何本身不看长度：每条回答只用实际生成、非 special 的 response token 计算四个 mean/std 特征，因此相似动力学的短回答和长回答具有可比较的 Q。prompt、padding、EOS 以及 tokenizer 声明的其他 special token 均不进入轨迹统计。

外层 loss 使用 response-token normalization：若第 i 条回答的 one-sided Huber 为 l_i，实际参与轨迹统计的生成 token 数为 T_i，则 `L = sum(T_i * l_i) / sum(T_i)`。T_i 是 stop-gradient 常数。这样 Q 不被长度污染，同时恢复 token-additive 的梯度质量；好侧回答即使很长，l_i 仍为零。代码同时按 micro-batch 和全部 FSDP rank 的总有效 token 数归一，不会因为动态分批或 GPU 数量改变 loss scale。vanilla OPD 保持其原始 token-mean。

run_qwen3-0.6b-base-slqp.sh 固定使用 slqp_only=true，因此正式目标不依赖 teacher、reference KL 或 GRPO group。当前正式方案只保证 pure-SLQP 的全局 response-token normalization；不要只把 slqp_only 改成 false 就声称完成 OPD+SLQP，联合目标还需要单独定义两种 loss 的相对尺度。

## 先为 0.6B student 做一次校准

不能直接复用其他 student 的质心、PCA 方向或边界。输入 JSONL 每行需要 quality_score，并使用以下两种格式之一：

    {"id":"x","prompt_token_ids":[1,2],"response_token_ids":[3,4],"quality_score":0.9}

或：

    {"id":"x","prompt":"题目文本","response":"回答文本","quality_score":0.9}

token ids 格式更可靠，因为它与实际训练轨迹完全一致。运行：

    cd verl
    python examples/g_opd/calibrate_slqp.py \
      --model Qwen/Qwen3-0.6B-Base \
      --input /path/to/labeled_responses.jsonl \
      --output /path/to/qwen3_0.6b_base_slqp.json \
      --batch-size 4

该步骤冻结四维均值/标准差、PCA 方向、两簇中心中点以及该 tokenizer 的 special-token 排除列表。quality_score 只用于确定 PCA 方向哪一侧代表高质量，不进入正式训练。请用当前脚本重新生成 version=2 校准文件；训练按校准文件中冻结的 token inclusion rule 执行，避免校准与训练口径不同。

## 重新收集 200 条逐 token hidden

run_collect_slqp_200.sh 分三个独立进程执行：

1. 用 vLLM 生成 200 条 Qwen3-0.6B-Base 数学回答。
2. vLLM 完全退出后，用 Hugging Face backbone replay；保存每个 response token 的最终层 hidden。
3. 打包 download.zip。

默认题目来自 guanning-ai/dapo14k，默认最大生成 4096 token。保存内容包括原题、prompt、回答文本、两侧 token IDs、prompt-end hidden、所有 response-token hidden、原始 response mean/last hidden、排除 special token 后的 trajectory mean/last hidden、trajectory mask、实际轨迹长度和四维轨迹特征。没有 stride，也没有文件大小硬上限。

服务器运行：

    cd verl
    export CUDA_VISIBLE_DEVICES=0
    bash examples/g_opd/run_collect_slqp_200.sh

若使用本地 parquet：

    DATA_FILE=/path/to/train.parquet bash examples/g_opd/run_collect_slqp_200.sh

完成后只需下载：

    verl/slqp_capture_200/download.zip

下载并解压后，对 responses.jsonl 的每个 id 补充 quality_score，形成 scored_responses.jsonl，然后在本地拟合：

    python examples/g_opd/fit_slqp_from_collected.py \
      --bundle /path/to/slqp_capture_200 \
      --scores /path/to/scored_responses.jsonl \
      --output /path/to/qwen3_0.6b_base_slqp.json

## 运行

vanilla OPD：

    cd verl
    bash examples/g_opd/run_qwen3-0.6b-base-vanilla-opd.sh

纯 SLQP：

    cd verl
    export SLQP_CALIBRATION=/path/to/qwen3_0.6b_base_slqp.json
    bash examples/g_opd/run_qwen3-0.6b-base-slqp.sh

数据路径、GPU 数、batch、最大回答长度和输出目录均可用脚本顶部同名环境变量覆盖。

两个正式训练脚本默认只训练和保存 checkpoint：`trainer.val_before_train=false`、`trainer.test_freq=-1`。OPD 与 SLQP 都完成后，再合并各自最终 checkpoint，使用 `math_eval/eval_math.py` 的同一参数统一测评，避免训练过程中的验证占用生成时间或造成两个方法测评时点不一致。

### 命令行类型与引号核对

- Hydra 的 `grpo`、`vllm`、`error`、`naive`、`icml_long` 和 experiment name 都是普通字符串；裸写 `key=value` 会被 OmegaConf 解析为字符串，不需要把引号字符本身传进配置。
- `trainer.logger='["console","wandb"]'` 是列表，不是字符串。外层 shell 单引号保证整个列表作为一个参数传给 Hydra。
- `VAL_FILES` 的默认值同样是 Hydra 列表文本，并通过 `data.val_files="$VAL_FILES"` 作为单个参数传入；不能把它改成未加括号的逗号分隔文本。
- 模型、数据、校准文件和输出路径全部使用双引号包住变量展开，因此路径里有空格不会被 shell 拆开。
- `true/false`、整数和浮点数按配置 schema 保持对应类型；没有把它们写成带引号的字符串。
- 收集脚本的 `--model`、`--data`、`--output` 是 argparse 字符串，变量展开已加双引号；`--samples`、长度、batch 和显存比例由 argparse 显式转换成 int/float。
- `SLQP_CALIBRATION` 是必填字符串路径；未设置时脚本会在启动训练前直接报错，不会静默使用错误默认值。

## 训练时必须观察

- actor/slqp_quality_mean
- actor/slqp_margin_mean
- actor/slqp_active_fraction
- actor/slqp_response_length_mean 与 actor/slqp_generated_tokens
- 四个 actor/slqp 特征
- actor/grad_norm
- monitor/train_accuracy_reward（训练集数学正确率监控，不参与 OPD/SLQP 优化）
- monitor/generated_response_tokens_mean
- monitor/correct_response_tokens_mean 与 monitor/incorrect_response_tokens_mean
- 数学验证集准确率与输出长度

如果 active fraction 很快降到零但数学准确率不升，说明 student 找到了校准几何的捷径；此时应停训并重新校准，而不是继续加大 loss weight。
