# Baseline 评测交付规范（CT-RATE Table1 / SOTA 口径）

目标：外部 baseline **不需要集成本体代码**，只要交付一份标准化的文本输出 `pred_jsonl`，我们就能在同一口径下产出：

- `pairs.jsonl`（`sample_id, method, pred_text, ref_text`）
- `extra_metrics_jsonl`（GREEN/RadCliQ/可选 proof 指标）
- `paper_metrics.json`（Table1 数字 + paired 显著性 + Holm）

本规范配合：
- `docs/external_baselines_adapter.md`（输出格式与转换脚本）
- `scripts/paper/run_table1_ct_rate.py`（一键驱动 Table1 口径）

---

## 0) 重要声明：什么时候可以写 “SOTA”

在没有拿到外部 baseline 的 **真实 test 输出**之前，不能写 “SOTA”。只能写：
- “我们引入了 proof-centric 可靠性维度，并在该维度上显著优于若干内部/弱基线”
- 或 “我们提供了可复现的统一评测入口，外部 baseline 对比待补”

要写 “SOTA on clinical metrics”，至少满足：
- 使用 **pp.md §6.1 所述官方 split**（或你能严格复现的等价 split），并在文中写清楚 split 来源/复现方式
- 对同一 test set，收集到每个 baseline 的完整输出（覆盖率 100%）
- clinical metrics（CheXbert/RadGraph/RadCliQ/GREEN/RaTEScore）版本/依赖写死（包含环境与模型权重版本）

---

## 1) 数据与 split（当前 dev vs 最终论文）

- **dev（当前 repo 默认）**：`/data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl` 的 `split=test`（目前是本地子集，便于快速迭代口径与脚本）。
- **最终论文**：必须切换到 `pp.md §6.1` 的官方/可复现 split（更大 test），否则不建议做 “SOTA” 主张。

---

## 2) Baseline 交付物（唯一必须）

每个 baseline 交付一份 UTF-8 JSONL（可多方法混在同一文件）：

```jsonl
{"sample_id":"<scan_hash>","method":"diallama","pred_text":"..."}
```

约束：
- `method`：方法名（会进入表格行名），例如 `ct2rep` / `diallama` / `mu2_llm`。
- `sample_id`：必须能对齐到 manifest 的 `scan_hash`。
- `pred_text`：自由文本（不要求 citations，不要求行格式）。

如果 baseline 输出的 id 不是 `scan_hash`：
1) 先把它转成 `pred_jsonl`（见下节），
2) 再用 `scripts/external/normalize_pred_jsonl_ids.py` 映射到 `scan_hash`。

---

## 3) 外部输出 → `pred_jsonl`（转换与 ID 对齐）

详见：`docs/external_baselines_adapter.md`。

常用命令（示例）：

```bash
# 1) 任意输出 → pred_jsonl
python scripts/external/to_pred_jsonl.py \
  --in /path/to/preds_dir \
  --format dir --glob '*.txt' \
  --method ct2rep \
  --out-jsonl /path/to/preds_ct2rep.jsonl

# 2) pred_jsonl 的 id 对齐到 manifest.scan_hash
python scripts/external/normalize_pred_jsonl_ids.py \
  --manifest /data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl \
  --in-pred-jsonl /path/to/preds_ct2rep.jsonl \
  --out-jsonl /path/to/preds_ct2rep_norm.jsonl
```

---

## 4) 一键产出 Table1 口径（推荐入口）

推荐把你现有的 ours `pairs.jsonl`（例如 E0214）与外部 baseline `pred_jsonl` 合并到同一个 Table1：

```bash
python scripts/paper/run_table1_ct_rate.py \
  --manifest /data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl \
  --split test \
  --base-pairs-jsonl outputs/E0214-ct_rate_rrg_dpo_full/pairs.jsonl \
  --external-pred-jsonl /path/to/preds_ct2rep_norm.jsonl /path/to/preds_diallama_norm.jsonl \
  --out-dir outputs/E0xxx-table1_ct_rate \
  --baseline-method fixed_grid \
  --holm-family all \
  --n-bootstrap 10000
```

可选（GREEN/RadCliQ/CheXbert via RadEval，Py3.11 环境）：

```bash
python scripts/paper/run_table1_ct_rate.py \
  --manifest /data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl \
  --split test \
  --base-pairs-jsonl outputs/E0214-ct_rate_rrg_dpo_full/pairs.jsonl \
  --external-pred-jsonl /path/to/preds_ct2rep_norm.jsonl \
  --out-dir outputs/E0xxx-table1_ct_rate \
  --run-radeval \
  --radeval-env /data/conda_envs/radeval311 \
  --baseline-method fixed_grid \
  --holm-family all \
  --n-bootstrap 10000
```

可选（把 proof 指标也并入 Table1；用于回答“可靠性 vs 临床指标”的同表对比）：

1) 导出 **ours** 的 proof/grounding 指标（来自 `run_baselines` 的 `baselines.json`）：

```bash
python scripts/paper/export_baselines_extra_metrics_jsonl.py \
  --baselines-json outputs/E0214-ct_rate_rrg_dpo_full/baselines_*/baselines.json \
  --manifest /data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl \
  --out-jsonl outputs/E0xxx-table1_ct_rate/extra_baselines.jsonl
```

2) 在同一次 Table1 里：
- 对外部 baseline 用 `--run-proof-external` 计算 wrapper proof 指标；
- 用 `--run-radeval` 计算 GREEN/RadCliQ/CheXbert；
- 用 `--extra-metrics-jsonl` 把 ours 的 proof 指标 merge 进来。

```bash
python scripts/paper/run_table1_ct_rate.py \
  --manifest /data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl \
  --split test \
  --base-pairs-jsonl outputs/E0214-ct_rate_rrg_dpo_full/pairs.jsonl \
  --external-pred-jsonl /path/to/preds_ct2rep_norm.jsonl \
  --out-dir outputs/E0xxx-table1_ct_rate \
  --run-radeval --radeval-env /data/conda_envs/radeval311 \
  --run-proof-external --tokenizer fixed_grid --budget-tokens 256 \
  --extra-metrics-jsonl outputs/E0xxx-table1_ct_rate/extra_baselines.jsonl \
  --baseline-method fixed_grid \
  --holm-family all \
  --n-bootstrap 10000
```

---

## 5) 评测公平性（pp.md §6.5 的落地写法）

- 外部 baseline 通常不输出 citations。
- 我们在 proof 指标上使用 **post-hoc citation wrapper**：不改变 baseline 的文本，只对生成文本解析出 frames 后附加 deterministic citations（用于审计）。
- 因此 proof 指标衡量的是 “文本在固定证据预算下是否能被审计性支持”，而非 “是否输出了某种格式”。

---

## 6) 产物清单（你交付/复现时要留档的文件）

在 `--out-dir` 下，至少应存在：
- `pairs_all.jsonl`：合并后的 `(sample_id, method, pred_text, ref_text)`（Table1 的唯一输入）
- `extra_radeval.jsonl`（如果启用）：GREEN/RadCliQ/CheXbert per-sample
- `extra_merged.jsonl`（如果存在多个 extra）：合并后的 extra metrics
- `paper_metrics.json`：Table1/显著性/CI 的最终 JSON（可再转成 latex 表格）
