# 外部 baselines 适配指南（pp.md §6.3）

目标：不把外部 baseline 本体代码集成进来，也能把它们的推理输出统一转成本仓库可评测的 `pred_jsonl`，并用统一口径跑 `pp.md` 的 proof 指标与（可选）NLG 指标。

## 1) `pred_jsonl` 输入格式（`eval_external_predictions` 需要）

每行一个 JSON：

```jsonl
{"sample_id":"<scan_hash>","method":"diallama","pred_text":"..."}
```

约束：
- `sample_id` 最好是 manifest 的 `scan_hash`。
- `method` 是方法名（用于聚合/分组）。
- `pred_text` 是生成的报告文本（自由文本，后续会用 `FrameExtractor` 抽取 finding 句并做 post-hoc citations）。

## 2) 外部输出 → `pred_jsonl`（通用转换脚本）

统一用：`scripts/external/to_pred_jsonl.py`

常见形态模板：

1) 目录下每例一个 `.txt`（文件名 stem 作为 id）

```bash
python scripts/external/to_pred_jsonl.py \
  --in /path/to/preds_dir \
  --format dir --glob '*.txt' \
  --method ct2rep \
  --out-jsonl /path/to/preds_ct2rep.jsonl
```

2) JSONL（每行一个 dict）

```bash
python scripts/external/to_pred_jsonl.py \
  --in /path/to/preds.jsonl \
  --format jsonl \
  --id-key scan_hash \
  --pred-key pred_text \
  --method diallama \
  --out-jsonl /path/to/preds_diallama.jsonl
```

3) JSON（list[dict] 或 dict[sid]=text）

```bash
python scripts/external/to_pred_jsonl.py \
  --in /path/to/preds.json \
  --format json \
  --id-key scan_hash \
  --pred-key pred_text \
  --method mvketr \
  --out-jsonl /path/to/preds_mvketr.jsonl
```

4) CSV/TSV（有表头或指定列号）

```bash
python scripts/external/to_pred_jsonl.py \
  --in /path/to/preds.csv \
  --format csv \
  --id-key scan_hash \
  --pred-key pred_text \
  --method 3d_ct_gptpp \
  --out-jsonl /path/to/preds_3d_ct_gptpp.jsonl
```

## 3) ID 对齐：不是 `scan_hash` 怎么办

如果 baseline 输出的 id 不是 manifest 的 `scan_hash`（常见是 `series_uid`/文件名/路径），先跑：

```bash
python scripts/external/normalize_pred_jsonl_ids.py \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --in-pred-jsonl /path/to/preds.jsonl \
  --out-jsonl /path/to/preds_norm.jsonl
```

该脚本会尽量把 `sample_id` 映射到 manifest 的 `scan_hash`（支持 `scan_hash/series_uid/volume stem/compute_scan_hash(patient_id,study_date,series_uid)`）。

## 4) 评测入口（pp.md §6.3）

```bash
python -m provetok.experiments.eval_external_predictions \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --split test \
  --resize-shape 64 64 64 \
  --pred-jsonl /path/to/preds_norm.jsonl \
  --tokenizer fixed_grid \
  --budget-tokens 256 \
  --output-dir ./outputs/E0xxx-external_eval
```

说明：
- proof 指标采用 post-hoc citations（pp.md §6.5）确保公平。
- 若 manifest 含 `mask_path`，会额外输出 grounding 指标。

## 5) baseline 级“最小模板”（需要你把实际输出路径/字段填进去）

这些模板不假设你用哪份开源实现，只约束“最终能转成 pred_jsonl”。

### CT2Rep
- 常见输出：每个 study 一个 txt。
- 模板：用 `--format dir`。

### Dia-LLaMA
- 常见输出：JSON/JSONL，字段可能是 `study_id`/`report`/`prediction`。
- 模板：`--format jsonl --id-key <你的字段> --pred-key <你的字段>`，必要时再 `normalize_pred_jsonl_ids.py`。

### MvKeTR
- 常见输出：CSV/JSONL。
- 模板：同上。

### 3D-CT-GPT++
- 常见输出：JSONL + 方法名。
- 模板：同上。

### μ²Tokenizer/μ²LLM
- 常见输出：JSONL（可能带多版本/多 seed）。
- 模板：先用 `--method mu2_tokenizer` 固定方法名，再按需要拆分不同 runs。

## 6) 自检：不依赖外部 baseline 的 smoke

当你还没拿到外部 baseline 输出时，可以用 manifest 直接生成一个 dummy `pred_jsonl` 来验证评测链路：

```bash
python scripts/external/make_dummy_pred_jsonl_from_manifest.py \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --split test --max-records 8 \
  --method dummy \
  --pred-source reference \
  --out-jsonl ./outputs/_dummy_preds.jsonl

python -m provetok.experiments.eval_external_predictions \
  --manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl \
  --split test \
  --resize-shape 64 64 64 \
  --max-samples 8 \
  --pred-jsonl ./outputs/_dummy_preds.jsonl \
  --tokenizer fixed_grid \
  --budget-tokens 256 \
  --output-dir ./outputs/_dummy_external_eval
```
