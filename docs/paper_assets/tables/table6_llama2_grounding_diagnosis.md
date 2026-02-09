# Table 6. Llama2PCG Polarity/Citation Diagnostics (Grounding Semantics)

- Curve: `/home/ubuntu/tiasha/ProveTok/outputs/E0172-backbone-llama2-mini-n57/baselines_curve_multiseed.json`
- Budgets: 2000000, 4000000, 7000000
- Seeds: [0, 1, 2]
- N: 57 (test samples)
- n_bootstrap(curve CI): 20000, CI: 0.95

**Grounding semantics**
- `IoU_pos_only`: 只统计 polarity∈{present,positive} 的 frames 上的 citations（见 `provetok/eval/metrics_grounding.py::_select_positive_citations`）。
- `IoU_all_frames` 仅用于诊断：把所有 frames 的 citations union 后算 IoU。由于 absent/negative statements 没有 lesion mask，对主结论不采用该口径。
- `abs` 统计口径：polarity∈{absent,negative} 的 frames 数量（用于解释“高预算回撤/IoU≈0 是否来自输出分布改变”）。

| Budget | fixed_grid: pos/total | fixed_grid: abs/total | fixed_grid: cite_pos/total | fixed_grid: cite_nonpos/total | fixed_grid: IoU_pos | fixed_grid: IoU_all | provetok_lesionness: pos/total | provetok_lesionness: abs/total | provetok_lesionness: cite_pos/total | provetok_lesionness: cite_nonpos/total | provetok_lesionness: IoU_pos | provetok_lesionness: IoU_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0000 | 0.0000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0000 | 0.0000 |
| 4000000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0000 | 0.0000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0008 | 0.0008 |
| 7000000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0000 | 0.0000 | 1/1 | 0/1 | 3/3 | 0/3 | 0.0000 | 0.0000 |

Notes: 表中为 mean 值；CI 详见曲线 JSON。若 `pos/total` 很低，则 `IoU_pos_only≈0` 可能是口径导致的自然结果；此时需要同时看 frame-level correctness 与 anti-silencing gates。
