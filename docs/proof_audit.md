# Proof Audit (2026-02-04)

本文件按 `docs/plan.md` 的 Claims (C####) 逐条检查：当前 `docs/experiment.md` + `.rd_queue/results/*.json` + `docs/results.md` 中的结果，是否足以 **证明** 每条 Claim 的 proof rule。

结论：截至 2026-02-04，本仓库闭环可跑，且 **C0001–C0006 均已通过** `python scripts/proof_check.py` 的判定。其中：
- **C0003（counterfactual non-triviality）已可按 plan 的最小 proof rule 证明**（`cite_swap` 显著击穿 unsupported，`no_cite` 显著击穿 grounding；见下文与 `python scripts/proof_check.py`）。
- **C0003 的 optional stronger check：non-oracle `omega_perm` 现在也可显著击穿 grounding**（见 E0157）。
- **C0001–C0006 已按当前仓库定义的 proof rule 全部通过**（见下文与 `python scripts/proof_check.py`）。更强版本（更严格 Pareto/latency-matched、跨数据集验证等）仍可在后续迭代补齐。

---

## C0001 — Pareto dominate in FLOPs/latency-matched

**Status:** Proved (paper-grade)

**Evidence checked**
- Paper-grade baselines curve (test): `outputs/E0138-full/baselines_curve_multiseed.json`
- Proof script: `python scripts/proof_check.py` (C0001)

**What is proved**
- 在 `B={2e6,3e6,4e6,5e6,6e6,7e6}`、`seeds>=5`、`n_bootstrap>=20000` 下，`provetok_lesionness` 相对 `fixed_grid` 的 `combined` 与 `iou_union` 均为 **正向且 one-sided paired bootstrap + Holm 后显著**（6/6 budgets 通过；见 `python scripts/proof_check.py` 的 C0001 明细）。
- 同一设置下还满足硬 gate：`warm_time_p95_s` 不超过 `fixed_grid` 的 +5%，且 `unsupported` 增幅不超过 +0.05（所有 budgets）。

**Remaining gaps (stronger version)**
- 可把 “Pareto dominate” 从当前的（quality + hard-gates）升级为更完整的多目标 Pareto（加入 overclaim/refusal/ECE + cold-start latency 等），并保持同协议对齐。

---

## C0002 — scaling law + allocation model, report regret

**Status:** Proved (paper-grade)

**Evidence checked**
- Paper-grade dev→test regret artifact: `outputs/E0141-full/fig3_results.json`
  - Inputs:
    - dev curve (split=val): `outputs/E0142-full/baselines_curve_multiseed.json`
    - test curve (split=test): `outputs/E0138-full/baselines_curve_multiseed.json`

**What is proved**
- 在 dev(split=val) 上，对每个候选 method 的 `combined(B)` 拟合 scaling law，并进行 AIC/BIC 模型选择；
- 在 test(split=test) 上，根据 dev 拟合预测每个 budget 的最优 method，并对比 test 上的 oracle 最优 method，输出 per-budget regret 曲线（见 `python scripts/proof_check.py` 的 C0002）。

---

## C0003 — counterfactual 显著击穿（paired bootstrap ≥10k + Holm）

**Status:** Proved (minimal proof rule)

**Evidence checked**
- Paper-grade counterfactual (rerun): `outputs/E0113-full/figX_counterfactual_20260203_181139/figX_counterfactual.json`
- (Older) Paper-grade counterfactual: `outputs/E0109-full/figX_counterfactual_20260203_140745/figX_counterfactual.json`

**Minimal proof rule (what is proved)**
- `cite_swap` 在 `unsupported_rate_cf_minus_orig` 上 **显著上升**（E0113：Holm 后 `p_value_holm=0.0`）。说明在 verifier 的 citation-relevance 口径下，swap 会被稳定检测到（Citations 不是“装饰”）。
- `no_cite` 在 `grounding_iou_union_orig_minus_cf` 上 **显著下降**（E0113：Holm 后 `p_value_holm=0.0`；`iou_max` 口径同样显著）。

**Optional stronger check (now proved)**
- **non-oracle `omega_perm` 现在可显著击穿 grounding（`iou_union`，且 margin 更足）**：`outputs/E0157-full/figX_counterfactual_20260204_202429/figX_counterfactual.json` 中 `grounding_iou_union_orig_minus_cf.omega_perm` 的 `p_value_holm=0.0044`（mean_diff≈0.0035）。
  - 该结果由 `E0135` 训练的 `saliency_cnn3d.pt` 重写 fixed-grid tokens 的 `score`（token score = cell 内 saliency 平均概率），并用 `ToyPCG(citation_strategy="score")` 选择 citations，从而强化 citations/Ω 与 lesion 的空间对齐，使 Ω-permutation 更“致命”。
  - 同一实验中 `iou_max` 口径也显著（`p_value_holm=0.0028`，mean_diff≈0.0070），不再卡边。

**Oracle sanity (not a paper claim, but a diagnostic)**
- `E0133` 使用 `--oracle-score`（GT mask 驱动 token.score 与 citations）时，`omega_perm` 对 grounding 可稳定显著击穿（Holm 后 `p_value_holm=0.0`）。这证明：当 citations/Ω 真正对齐时，Ω-permutation 会“致命”；而当前系统未通过的原因更可能在于 citations/Ω 对齐强度不足，而非统计脚本本身。

**Required next steps**
- （可选）若要把 `omega_perm` 提升为必选 proof rule：建议做 multi-seed / 更大数据集复验，并锁定更稳健的主指标（目前 `iou_union` 已通过但较贴近阈值）。
- （可选）重新设计 evidence-drop（例如只 drop 被 cite 的 token、或 drop top-attention tokens），并把 correctness 指标（frame_f1 vs GT）纳入 paired bootstrap，以增强“反事实击穿”的覆盖面与解释力。
- 详细分析与备选方案见：`docs/proof_failure_analysis.md`。

---

## C0004 — ReX pixel-level grounding 显著提升

**Status:** Proved (paper-grade)

**Evidence checked**
- Paper-grade grounding proof (mini, paired bootstrap + Holm, multi-budget): `outputs/E0143-full/figX_grounding_proof.json`
- (Stronger supplement) Grounding proof on 100g via saliency-scored citations: `outputs/E0156-grounding_proof_100g_saliency_full/figX_grounding_proof.json`

**What is proved**
- 在 `B={2e6,3e6,4e6,5e6,6e6,7e6}`、`seeds>=5`、`n_bootstrap>=20000` 下，`provetok_lesionness` 的 `iou_union` 相对 `fixed_grid` 与 `roi_variance` 均为 **正向且 one-sided paired bootstrap + Holm 后显著**（6/6 budgets 通过；见 `python scripts/proof_check.py` 的 C0004）。
- 在更大数据集 `rexgroundingct_100g` 上，使用 `SaliencyCNN3D` 重写 token.score 后，也能在相同预算/统计口径下对 `fixed_grid` 与 `roi_variance` 均达到 **6/6 budgets PASS**（见 E0156）。

**Notes (key fix that enabled proof)**
- toy token embedding 过去依赖 experiment seed，导致 learned lesionness head 在多 seed 下失效；现已改为 **seed-invariant toy embedding**（更接近真实 encoder），并配合 `ToyPCG(citation_strategy=score_interleave)` 与 level3 lesionness head，使 grounding 信号足够强。

---

## C0005 — refusal calibration（unsupported↓ 且 critical miss-rate 不升）

**Status:** Proved (paper-grade)

**Evidence checked**
- Paper-grade refusal calibration (dev→test, fixed τ across budgets): `outputs/E0144-full/figX_refusal_calibration.json`

**What is proved**
- 在 dev(split=val) 上校准一次 `tau_refuse`，并在 test(split=test) 的多预算上固定复用。
- test 上 `critical_miss_rate <= δ`（δ=0.05），并满足 refusal 的 hard gates（`refusal_ece<=0.15`、`refusal_rate<=0.20`）；同时 `unsupported_rate` 相对 no-refusal baseline 在 ≥4/6 budgets 下降（本次为 6/6；见 `python scripts/proof_check.py` 的 C0005）。

**Notes (key fix that enabled proof)**
- 之前 calibration 只改 `generation.refusal`，但 **未重跑 verifier**，导致 unsupported 口径对 τ 不敏感；现已在校准与评测阶段对每个 τ 重新计算 verifier issues。

---

## C0006 — baseline completeness + cost accounting + matched 不缺席

**Status:** Proved (paper-grade)

**Evidence checked**
- Paper-grade baselines curve (test): `outputs/E0138-full/baselines_curve_multiseed.json`
- Proof script: `python scripts/proof_check.py` (C0006)

**What is proved**
- baselines curve artifact 中包含 `fixed_grid/slice_2d/slice_2p5d/roi_crop/roi_variance/ct2rep_strong` 等 required methods；
- `costs_json` + `budgets_by_method` 存在（含 ROI selector 额外成本入账），可审计复现 matched budget_targets；
- `ct2rep_strong` weights 可复现且非退化（最后一个 budget 的 `frame_f1(mean) >= 0.05`；见 `python scripts/proof_check.py` 的 C0006）。

**Remaining gaps (stronger version)**
- “强 3D RRG baseline” 仍建议用公开可复现实现替代 `ct2rep_like` 占位；
- 完整的 latency-matched（baseline 同协议 warm/cold mean + P95）与跨域（CT-3DRRG）闭环仍建议补齐。
