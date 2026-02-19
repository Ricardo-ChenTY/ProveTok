# Plan

## Goals

以 `pp.md` 作为当前 proposal/spec，收敛到一条可复现的“论文级 Table1 + 外部 baseline + proof 指标”主线，并移除旧路线（FLOPs/scaling/旧合成任务）在执行层面的混排与术语混淆。

当前范围（保留）：
- CT-RATE（RRG-DPO 可比预处理）+ 内部方法变体 + 外部 baseline（先以 CT2Rep 为可跑强基线）
- 统一 Table1 口径：NLG/clinical/structured/LLM metrics + proof-centric metrics（WeightedIssue/R1–R4）
- 外部 baselines 统一输入：`pred_jsonl` + post-hoc citation wrapper（audit-only，不改文本）

明确不做（deferred）：
- `pp.md §6.1` 的 CT-RATE official RRG-DPO split 复现（你已明确暂缓）

## Glossary (避免术语混淆)

- **RRG-DPO 预处理**：把 CT-RATE 体数据统一到 `0.75×0.75×1.5mm³` spacing，并 `480×480×240` crop/pad 后落盘（见 `scripts/data/preprocess_manifest_rrg_dpo.py`）。
- **`--resize-shape`**：dataloader 额外做的下采样/重采样（为了跑得动）。它会改变有效分辨率，和“RRG-DPO 预处理已可比”不是一回事。论文级对比要么禁用/对齐它，要么在表格标题里明确写成“downsampled setting”。
- **Token score model（打分/定位）**：决定“引用哪些 token/先 split 哪些 cell”的信号源，当前主要通过 `--saliency-weights ...`（UNet/ResUNet/VNet/CNN3D 等）注入到 token.score（见 `provetok/models/saliency*.py`、`provetok/baselines/tokenizers.py`）。
- **Token embedding encoder（特征编码器）**：`TokenEncoder(encoder=...)` 用来生成 token.embedding 的 3D encoder feature map（见 `provetok/bet/tokenize.py`）。目前 repo 内可用的是 `SimpleCNN3D`（随机初始化轻量 encoder，用于替代 toy patch embedding）；CT-CLIP/RadFM 级别的预训练 encoder 尚未接入。

## Claims (C####)

- [ ] C0101: CT-RATE（RRG-DPO 预处理 test split）Table1 可复现产出
  - Evidence: E0214, E0226, E0227, E0228, E0229, E0230
  - Proof rule:
    - `scripts/paper/run_table1_ct_rate.py` 在 `n=58` test 上产出 `paper_metrics.json`，包含：BLEU/METEOR/ROUGE、CheXbert、RadGraph、RadCliQ、GREEN、RaTEScore、WeightedIssue@B=256，以及 paired 统计与 Holm 校正。
    - 对外部 baseline（CT2Rep）额外报告 proof 指标时，必须走 post-hoc citation wrapper（audit-only，不改文本），并保证 100% 覆盖 test split。
  - Notes: 现阶段“SOTA”只允许写成 **proof-centric reliability under budget**；文本指标只做 comparability/不退化声明，避免不公平设定。

- [ ] C0102: 外部 baseline 评测协议可扩展且不会引入不可复现依赖漂移
  - Evidence: E0230 + `docs/external_baselines_adapter.md`
  - Proof rule:
    - 任意外部方法只需提供 `pred_jsonl {sample_id, method, pred_text}`，即可进入统一 Table1 driver；heavy 指标（GREEN/RadCliQ/CheXbert）通过隔离环境计算并落盘 `extra_metrics_jsonl`（或由一键脚本封装），保证可复现。

## Plan Items (P####)

- [ ] P0101: 清理执行文档，只保留当前 pp.md 主线
  - Linked claims: C0101, C0102
  - Definition of done:
    - `docs/plan.md` / `docs/mohu.md` / `docs/experiment.md` 不再混排旧路线（FLOPs/scaling/早期合成任务）；
    - 明确区分 `--saliency-weights`（score）与 `--encoder`（embedding encoder）；
    - legacy 内容保留在 `docs/*_legacy_20260219.md`。
  - Verification: `python scripts/rd_queue.py make --stage full --next --out .rd_queue/_verify_queue.json`
  - Touchpoints: `docs/plan.md`, `docs/mohu.md`, `docs/experiment.md`

- [ ] P0102: 跑通 CT2Rep full + Table1 full，并同步台账
  - Linked claims: C0101
  - Definition of done:
    - `E0228 full` 产出 `/data/provetok_runs/ct2rep_ct_rate_100g_rrg_dpo_full/model_best.pth`；
    - `E0229 full` 产出 `outputs/E0229-ct2rep_pred_full/preds_ct2rep.jsonl`（覆盖 test=58）；
    - `E0230 full` 产出 `outputs/E0230-table1_with_ct2rep_full/paper_metrics.json`。
  - Verification: `python scripts/rd_queue.py sync`
  - Touchpoints: `scripts/external/train_ct2rep_baseline.py`, `scripts/external/infer_ct2rep_to_pred_jsonl.py`, `scripts/paper/run_table1_ct_rate.py`

- [ ] P0103: 解决“可比预处理 vs resize_shape”的口径混淆
  - Linked claims: C0101
  - Definition of done:
    - `docs/experiment.md` 的每条 CT-RATE 相关实验明确写清是否额外使用 `--resize-shape`；
    - Table1 标题/脚注明确当前 setting（full-res vs downsampled），避免 reviewer 认为不公平。
  - Verification: `rg -n "resize-shape" docs/experiment.md && rg -n "downsample" docs/experiment.md`
  - Touchpoints: `provetok/data/dataset.py`, `provetok/experiments/run_baselines.py`, `scripts/paper/run_table1_ct_rate.py`

## Changelog

- 2026-02-19: 归档旧版执行文档到 `docs/plan_legacy_20260219.md` / `docs/experiment_legacy_20260219.md` / `docs/mohu_legacy_20260219.md`，主文件开始收敛到 `pp.md` Table1 主线。
