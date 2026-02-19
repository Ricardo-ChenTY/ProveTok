# RadEval 环境（用于 GREEN / RadCliQ 计算）

本仓库的主环境是 Python 3.13（为了工程/训练管线方便），但多数论文级文本评测（尤其是 GREEN / RadCliQ / RadGraph）在 Py3.13 上依赖链不稳定。

因此我们建议用 **单独的 conda 环境（Py3.11）** 跑 RadEval，然后把结果导出为 `extra_metrics_jsonl`，再交给本仓库的 `scripts/paper/compute_paper_metrics.py` 做合并与统计。

## 1) 创建环境（推荐）

```bash
conda create -n radeval311 -y python=3.11
conda activate radeval311
python -m pip install -U pip
```

## 2) 安装 RadEval（以及必要依赖）

RadEval 官方依赖版本可能会 pin 到你机器上暂时不可用的 torch 版本；实践里更稳的是：
1) 先装你环境可用的 `torch/transformers`，
2) 再用 `--no-deps` 安装 RadEval 本体。

示例（CUDA 机器）：

```bash
# 先装 torch（按你的 CUDA/平台选择合适 index-url）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 评测依赖（版本不强 pin，尽量让 pip 选到兼容 Py3.11 的 wheel）
python -m pip install transformers rouge-score bert-score scikit-learn 'numpy<2' medspacy stanza pillow sentencepiece datasets opencv-python matplotlib accelerate pandas

# RadGraph（RadCliQ v1 依赖 RadGraph）
python -m pip install radgraph

# RadEval 本体（从 GitHub 安装，避免 PyPI 旧版本/构建问题）
python -m pip install --no-deps git+https://github.com/jbdel/RadEval.git
```

## 3) 生成 extra_metrics_jsonl（GREEN/RadCliQ）

在 RadEval 环境中运行：

```bash
python scripts/external/compute_radeval_metrics_jsonl.py \
  --text-pairs-jsonl <pairs.jsonl> \
  --out-jsonl <extra_metrics.jsonl>
```

## 4) 回到本仓库环境合并 + 统计

```bash
python scripts/paper/compute_paper_metrics.py \
  --text-pairs-jsonl <pairs.jsonl> \
  --extra-metrics-jsonl <extra_metrics.jsonl> \
  --baseline-method <baseline> \
  --holm-family all \
  --out <paper_metrics.json>
```

