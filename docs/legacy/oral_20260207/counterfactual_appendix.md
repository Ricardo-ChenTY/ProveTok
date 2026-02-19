# Counterfactual Appendix (Definitions + Statistics)

本附录固定 counterfactual 的**定义**与**统计判定口径**，避免“后验改实验/改统计”。

> Source of truth: `provetok/experiments/figX_counterfactual.py`

## 1) Counterfactual definitions

对每个样本先生成原始结果 `(tokens, gen, issues)`，然后构造以下反事实（记作 `cf`）：

- **`omega_perm`**（Ω-permutation）：置换 tokens 的 `cell_id`（Ω），保持 embedding 不变；再用同一 `gen` 计算 grounding（等价于“引用到错区域”）。
- **`token_perm`**（token-permutation）：置换 tokens 的 embedding，保持 `cell_id` 不变；再生成/评测（等价于“证据内容被打乱”）。
- **`cite_swap`**（citation swap）：在同一报告内交换不同 frame 的 citations（保持每个 frame 的引用长度分布），再评测 unsupported/grounding/correctness。
- **`evidence_drop`**（drop cited tokens）：删除被引用的 tokens（drop 原 gen 的 citations 对应 token_ids），再用 PCG 重新生成并评测（等价于“拿走关键证据”）。
- **`no_cite`**（remove citations）：保持 frames 不变，清空所有 citations（等价于“断言不携带证据”）。

## 2) Metrics

### 2.1 Grounding
- `iou_union`: 把（正向 frame 的）citations union 成一个 3D mask，与 lesion_union mask 做 IoU。
- `iou_max`: 单个 citation mask 与 lesion_union 的最大 IoU（更偏“最佳命中”口径）。

### 2.2 Verifier-backed trust
- `unsupported_rate`（primary）：使用 **relevance-only** verifier（U1.4）计算 unsupported，避免 coverage/uncertainty confound。
- `unsupported_rate_full` / `overclaim_rate`：使用完整规则集作为分析项（不是 primary proof 口径）。

### 2.3 Correctness / Text (optional, enabled when deps exist)
- `frame_f1`: generation frames vs GT frames 的 Hungarian matching F1。
- `bleu`, `rougeL`: generation text vs GT report_text（当 `sacrebleu`/`rouge-score` 可用时启用）。

## 3) Statistics (paired bootstrap + Holm)

对每个 counterfactual key（`omega_perm/token_perm/cite_swap/evidence_drop/no_cite`）做 paired bootstrap：

- Grounding 与 correctness/text：用 **`orig - cf`** 作为差值（“被击穿”应表现为 `mean_diff > 0`）。
- Issue rates：用 **`cf - orig`** 作为差值（“变更导致更糟”应表现为 `mean_diff > 0`）。

统计量：
- `mean_diff`：样本级差值的均值
- `ci_low/ci_high`：bootstrap CI（默认 95%）
- `p_value`：bootstrap sign test 的 **two-sided** p-value（见 `provetok/eval/stats.py::paired_bootstrap_mean_diff`）
- `p_value_holm`：对同一组 counterfactual keys 的 p-values 做 Holm-Bonferroni 校正

**显著性判定**（paper/oral 口径）：在对应 metric 上 `p_value_holm < 0.05` 且 `mean_diff` 符合预期方向。

