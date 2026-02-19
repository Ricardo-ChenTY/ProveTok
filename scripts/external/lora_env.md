# LoRA/PEFT 环境（pp.md §6.6）

本仓库主环境是 Python 3.13；但 LLM 训练/PEFT 依赖链在 Py3.13 上更容易出问题。

因此我们沿用 **Py3.11 的 conda 环境** 来跑 LoRA smoke（可以复用 `radeval311`）。

## 0) 环境位置

- 推荐 env：`/data/conda_envs/radeval311`（Python 3.11）

验证：

```bash
conda run -p /data/conda_envs/radeval311 python -V
```

## 1) 安装 PEFT

```bash
conda run -p /data/conda_envs/radeval311 python -m pip install -U peft
```

验证：

```bash
conda run -p /data/conda_envs/radeval311 python -c "import peft; print(peft.__version__)"
```

## 2) LoRA smoke 路径（数据构建→训练→端到端验证）

### 2.1 构建 SFT JSONL（小样本）

```bash
conda run -p /data/conda_envs/radeval311 python scripts/external/build_lora_sft_jsonl_from_manifest.py \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --split train \
  --max-records 2 \
  --resize-shape 64 64 64 \
  --max-tokens-in-prompt 16 \
  --max-frames 1 \
  --contract-mode schema_only \
  --out-jsonl outputs/E0223-lora_sft_smoke/sft.jsonl \
  --out-meta outputs/E0223-lora_sft_smoke/meta.json
```

### 2.2 训练 LoRA（Smoke：少量 steps）

```bash
HF_HOME=/data/hf_cache conda run -p /data/conda_envs/radeval311 python scripts/external/train_lora_pcg_sft.py \
  --model-path /data/models/Llama-2-7b-chat-hf \
  --train-jsonl outputs/E0223-lora_sft_smoke/sft.jsonl \
  --output-dir outputs/E0224-lora_sft_train_smoke \
  --device cuda \
  --dtype float16 \
  --max-steps 2 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-len 1024
```

### 2.3 端到端 agent smoke（加载 adapter）

```bash
HF_HOME=/data/hf_cache conda run -p /data/conda_envs/radeval311 python scripts/external/smoke_lora_agent.py \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --split test \
  --resize-shape 64 64 64 \
  --model-path /data/models/Llama-2-7b-chat-hf \
  --lora-adapter outputs/E0224-lora_sft_train_smoke \
  --device cuda \
  --dtype float16 \
  --contract-mode schema_only \
  --max-frames 1 \
  --max-new-tokens 220 \
  --budget-tokens 64 \
  --max-steps-per-finding 1 \
  --output-dir outputs/E0225-lora_agent_smoke
```
