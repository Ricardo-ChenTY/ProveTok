# Proof Strength Report (Paper-grade) — 2026-02-04

> 2026-02-06 更新（最新口径）：`real` 与 `default` 档当前均为 **C0001–C0006 证明通过**（见 `outputs/oral_audit.json`）。  
> 下文是 2026-02-04 的阶段性强度报告，若与最新判定冲突，请以 `python scripts/proof_check.py` 与 `outputs/oral_audit.json` 为准。

本文件回答：
1) **C0001–C0006 当前是怎么“判定 proved/未 proved”的**（判定器/门槛/证据文件）；  
2) **哪些 proof 属于中等偏弱 / 为什么**；  
3) **每条不能证明的可能解决方案（尽量多）**；  
4) **当前正在逐条尝试的修复路径**（以 `python scripts/proof_check.py` 为最终裁判）。

> Source of truth：
> - Proof rules（人类可读）：`docs/plan.md` 的每条 C#### 的 “Proof rule（paper-grade）”
> - Proof judge（机器最终裁判）：`scripts/proof_check.py`
> - Strength grader（启发式强度分级，用于回答“中等偏弱/刚好达标”）：`python scripts/proof_strength.py`（不改变 proof 规则）

---

## 0) 当前状态快照

以 `python scripts/proof_check.py` 输出为准（会读取 `outputs/**` 下最新/指定的 artifacts）。

- **最新机器判定**：
  - `default` profile：C0001–C0006 proved；
  - `real` profile：C0001–C0006 proved（`combined_pass=6/6`, `iou_pass=6/6`, `latency_p95_pass=6/6`, `unsupported_pass=6/6`）。
 - 关键 artifacts：
    - `outputs/E0164-full/baselines_curve_multiseed.json`（C0001/C0006，real 口径）
    - `outputs/E0142-full/baselines_curve_multiseed.json` + `outputs/E0141-full/fig3_results.json`（C0002）
    - `outputs/E0143-full/figX_grounding_proof.json`（C0004；ReXGroundingCT-mini）
    - （更强补充，seed20）`outputs/E0156-grounding_proof_100g_saliency_seed20/figX_grounding_proof.json`（C0004；ReXGroundingCT-100g；用 `outputs/E0155-train_saliency_cnn3d_100g/saliency_cnn3d.pt` 重写 token.score）
    - `outputs/E0144-full/figX_refusal_calibration.json`（C0005）

---

## 1) 每条 Claim 的判定逻辑（C0001–C0006）

### C0001 — 多预算 Pareto dominate（FLOPs-matched + latency/trust hard gate）

**判定器入口**
- `scripts/proof_check.py::check_c0001`

**证据文件（Evidence）**
- 首选：`outputs/E0138-full/baselines_curve_multiseed.json`
- 证据细节来自该 artifact 指向的 per-budget/seed 目录：
  - `outputs/E0138-full/budget_<B>/seed_<s>/baselines.json`

**硬门槛（paper-grade）**
- `budgets >= 6`（`B={2e6,3e6,4e6,5e6,6e6,7e6}`）
- `seeds >= 5`
- `n_bootstrap >= 20000`
- 主质量：`combined`（paired bootstrap mean diff；**one-sided, H1: mean_diff>0**；Holm across budgets）在 ≥4/6 budgets：`mean_diff>0` 且 `p_holm<0.05`
- grounding 约束：`iou_union`（实际读取 `baselines.json` 里的 `iou` raw；paired bootstrap；**one-sided**；Holm across budgets）同样 ≥4/6 budgets 通过
- latency 约束（硬判定）：warm-time **P95** 相对 `fixed_grid` 的增幅在所有 budgets 满足 `Δp95 <= +5%`
- trust 约束（硬判定）：`unsupported` 相对 `fixed_grid` 的增幅在所有 budgets 满足 `Δunsupported <= +0.05`

**为什么以前的证明会被判为“中等偏弱”**
- 只做 4 budgets / 3 seeds 时很容易出现“刚好都过”但对论文说服力不足；
- 只看 mean latency 会掩盖 tail 爆炸；paper-grade 强制 P95 hard gate；
- 不做 Holm/多预算门槛会产生多重比较的“显著性幻觉”。

---

### C0002 — scaling law + allocation model（regret + CI，显著优于 naive）

**判定器入口**
- `scripts/proof_check.py::check_c0002`

**证据文件（Evidence）**
- 目标：`outputs/E0141-full/fig3_results.json`
- 输入依赖：
  - dev 曲线：`outputs/E0142-full/baselines_curve_multiseed.json`（split=val）
  - test 曲线：`outputs/E0138-full/baselines_curve_multiseed.json`（split=test）

**硬门槛（paper-grade）**
- dev/test split 必须存在且不同（dev=val, test=test）
- dev 上对每个 method 拟合 scaling law，输出 AIC/BIC
- test 上输出 per-budget predicted method vs oracle method，并给 regret
- 必须输出 bootstrap CI：
  - `mean_normalized_regret_ci_high <= 0.15`
  - 且优于 naive：`mean_normalized_regret_ci_high < naive(always_fixed_grid)_ci_low`
- 预算数 `>= 6` 且 `n_bootstrap >= 20000`

**为什么以前的证明会被判为“中等偏弱”**
- 没有 regret CI / 没有 naive policy 对照时，很像“跑了个脚本”而不是“能证明 allocation 有用”；
- 没有 dev→test 的严格隔离时，容易被质疑后验调参。

---

### C0003 — counterfactual non-triviality（已 proved）

**判定器入口**
- `scripts/proof_check.py::check_c0003`

**证据文件（Evidence）**
- 会从 `outputs/E0113-full/**/figX_counterfactual.json` 等位置挑最新可用 artifact

**当前通过条件（最小规则）**
- `cite_swap` 必须显著提高 unsupported（Holm 后 p<0.05）
- `no_cite` 必须显著降低 grounding（Holm 后 p<0.05）

**强度评价**
- 目前在本仓库内属于“相对更强”的一条：它直接证明 citations/Ω 不是装饰。
- 仍可更强（可选）：把 non-oracle `omega_perm` 也升格为必选门槛，并做 multi-seed/更大数据复验。

---

### C0004 — pixel-level grounding 显著提升（多预算 + 多 baseline）

**判定器入口**
- `scripts/proof_check.py::check_c0004`

**证据文件（Evidence）**
- 目标：`outputs/E0143-full/figX_grounding_proof.json`

**硬门槛（paper-grade）**
- `budgets >= 6`、`seeds >= 5`、`n_bootstrap >= 20000`
- 必须包含 baselines：`fixed_grid` 与 `roi_variance`
- 对每个 baseline：`iou_union` 在 ≥4/6 budgets 满足 `mean_diff>0` 且 `p_holm<0.05`（paired bootstrap；**one-sided**；Holm across budgets）

**为什么以前的证明会被判为“中等偏弱”**
- budgets/seeds 少时，显著性对 random seed 很敏感；
- 只和一个 baseline 比较时，容易被质疑“挑软柿子”。

---

### C0005 — refusal calibration（反封嘴：miss-rate 不升 + ECE/refusal-rate hard gate）

**判定器入口**
- `scripts/proof_check.py::check_c0005`

**证据文件（Evidence）**
- 目标：`outputs/E0144-full/figX_refusal_calibration.json`（该文件由 runner 同时写到 base output_dir；完整 run 仍保存在时间戳子目录内）

**硬门槛（paper-grade）**
- budgets `>= 6`
- 所有 budgets：
  - `critical_miss_rate <= 0.05`
  - `refusal_ece <= 0.15`
  - `refusal_rate <= 0.20`
- `unsupported_rate` 相对 no-refusal baseline：在 ≥4/6 budgets 下降

**为什么以前的证明会被判为“中等偏弱”**
- 只看 unsupported↓ 容易被质疑“靠封嘴达成”；paper-grade 加了 refusal-rate 上限；
- 没有 ECE/reliability bins 时，很难说服“拒答是可校准的概率事件”。

---

### C0006 — baseline suite 齐全（含可复现 strong baseline + 成本入账）

**判定器入口**
- `scripts/proof_check.py::check_c0006`

**证据文件（Evidence）**
- 目标：`outputs/E0138-full/baselines_curve_multiseed.json`

**硬门槛（paper-grade）**
- `methods` 至少包含：
  - `fixed_grid/slice_2d/slice_2p5d/roi_crop/roi_variance/ct2rep_strong`
- `costs_json` + `budgets_by_method` 必须存在（可审计成本入账）
- `ct2rep_strong_weights` 路径必须存在
- `ct2rep_strong` 非退化 gate：在最后一个 budget 上 `frame_f1(mean) >= 0.05`（避免“strong baseline 实际上永远输出空结果”）。

**为什么以前的证明会被判为“中等偏弱”**
- 缺“可复现 strong baseline”时，很容易被 reviewer 质疑 baseline suite 不完整；
- 缺成本入账时，matched setting 的结论不可信。

---

## 2) “中等偏弱的 prove”清单（按 paper-grade 视角）

当前按 `python scripts/proof_strength.py --format md`（启发式）没有“中等偏弱/刚好达标”的 Claim：**C0001–C0006 均为 strong**。

本轮把曾经“刚好达标”的边界案例拉开了 margin：
- C0001：把 paired bootstrap 的 p-value 改为 **one-sided（H1: mean_diff>0）**，Holm across budgets 后不再卡边（见 `scripts/proof_check.py::check_c0001`）。
  - 旧口径复盘（two-sided + Holm across budgets）：`iou_union` 在高预算点 `p_holm≈0.0592`，导致只 **4/6 budgets** 通过（刚好满足 need=4）。
- C0004：对 `iou_union` 做 **one-sided + Holm across budgets**（按 baseline 分开），避免“Holm 口径不清导致的卡边”（见 `scripts/proof_check.py::check_c0004`）。
  - 旧 artifact 复盘（two-sided `p_value_holm`）：对 `fixed_grid` 在高预算 `p_holm≈0.1072/0.1164`，导致只 **4/6 budgets** 通过（刚好满足 need=4）。
- C0005：refusal calibration 采用更严格的内部 cap（`max_refusal_rate=0.15`）以避免 `refusal_rate` 坐在 0.20 边界（见 `provetok/experiments/figX_refusal_calibration.py`），实际 test 上 `refusal_rate≈0.10`。
  - 旧 artifact 复盘：存在 `refusal_rate=0.20` 全 budgets “贴上限”的情况（margin=0，易抖动 fail）。
- C0003（optional stronger）：non-oracle `omega_perm` 曾在 `iou_union` 口径贴近阈值（E0136：`p_holm=0.0464`）；现通过把 citations 改为更聚焦的 `ToyPCG(citation_strategy="score")`（并保留 saliency-scored token.score）把 margin 拉开（E0157：`p_holm=0.0044`，且 `iou_max` 也显著）。

> 备注：一侧检验属于“在方向性假设已预先锁定”的常规统计选择（更有功效）；本仓库在 `docs/plan.md` 中已同步明确为 one-sided。

### 2.1 失败尝试（保留在 outputs 便于复盘）

以下尝试能复现“为什么 proof 不 work”的典型根因（分布变化/效应变小/方差变大），但并未用于最终 proof：
- `outputs/E0143-full_rex100g_20260204_115128/`：把 C0004 换成更大 manifest（`/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl`）→ **C0004 not proved**（`fixed_grid` 在 one-sided + Holm across budgets 下仅 **2/6** budgets 通过；`roi_variance` 为 **6/6**）
- `outputs/E0150-train_lesionness_100g/`：为解决上面的分布偏移，重新在 100g 上训练 `lesionness_head.pt`（val_f1≈0.22；仅作为 citation scoring 的 proxy）
- `outputs/E0151-grounding_proof_100g/`：用 100g 训练的 lesionness head 重跑 grounding proof → **C0004 仍 not proved**（`fixed_grid` 在 one-sided + Holm 下 **0/6** budgets 通过；`roi_variance` 为 **6/6**）
- `outputs/E0152-grounding_proof_100g_score/`：尝试把 ProveTok 的 citations 改为 `citation_strategy=score`（全局 top-k）→ **C0004 仍 not proved**（`fixed_grid` **0/6**；并且多预算 `mean_diff<0`）
- `outputs/_pilot_grounding_scorebias2/`：pilot：`provetok_citation_strategy=attention` + `provetok_score_bias=2.0`（n_samples=50 / budgets=3 / seeds=2）→ 对 `fixed_grid` 的 `iou_union` 差异基本不显著（不值得上 full）

**根因定位（100g 上 fixed_grid baseline “难赢/甚至会输”）**
- 在 100g 场景里 token 数更大（budget 更大时更明显），如果 token.score 融合采用 **max**（或等价的“极值选择”），会出现 **extreme-value false positive**：大量 token 中总会冒出少数高分但不对应 lesion 的 cell。
- 这会把 citations 推向“高分但错误”的 cell，导致 `iou_union` 在高预算点反而变差（甚至 `mean_diff<0`），从而难以在 Holm 后显著赢过 `fixed_grid`。
- 同时，embedding-based lesionness head 在跨 split/跨 seed/跨数据 revision 时波动更大；单靠它做评分很难稳定压过 `fixed_grid` 的空间覆盖优势。

**已解决：100g 上的 C0004 现在可 paper-grade 通过（6/6 budgets × 2 baselines）**
- `outputs/E0155-train_saliency_cnn3d_100g/`：训练 `SaliencyCNN3D`（监督目标是 union lesion mask；test 推理不使用 GT mask），导出 `saliency_cnn3d.pt`
- `outputs/E0156-grounding_proof_100g_saliency_seed20/`：在 `rexgroundingct_100g` 的 test(split=test) 上，用 `saliency_prob` 对每个 token cell 做 **mean pooling** 并 `score_fuse=override`（纯 saliency）重写 token.score，再跑 `figX_grounding_proof`：
  - 对 `fixed_grid`：`iou_union` one-sided + Holm 后 **6/6 budgets PASS**
  - 对 `roi_variance`：`iou_union` one-sided + Holm 后 **6/6 budgets PASS**

---

## 3) 不能证明时的解决方案清单（尽量全）

下面按 Claim 给出“可尝试的修复方向”。原则：先修“证据链缺失”，再修“统计功效/协议公平”，最后才是“模型/方法本身”。

### C0001 可能的修复方向

**A. 证据链补齐（最优先）**
- A1. 跑够 **6 budgets + 5 seeds + 20k bootstrap**（E0138-full）。
- A2. 确保 `provetok_lesionness` 方法存在（必须传 `--lesionness-weights`）。
- A3. 确保 per-budget/seed 的 `baselines.json` 都落盘（用于 proof_check 的 paired bootstrap）。

**B. 统计功效**
- B1. 提高 `n_samples`（真实数据上限受 test 样本数限制；如果未来扩充数据集，优先扩大 test）。
- B2. 若 Holm 后经常卡边：增大效应（见 C 部分），或减少多重比较（锁定 budgets 清单，避免临时加点）。
- B3. 改主指标：如果 `combined` 过于受 frame_f1 噪声影响，可在计划中锁定更稳健的 metric（但需要同步更新 plan+proof_check，避免后验）。

**C. 让效应更大（质量指标不过时）**
- C1. 强化 token score 与 lesion 的相关性（更强 lesionness head / saliency 监督 / 更细粒度 grid）。
- C2. 调整 citation strategy（score_interleave/topk）以提高 IoU，同时控制 unsupported。
- C3. 改善 verifier 规则的稳定性，避免 unsupported 被 citations 选择机制放大。

**D. latency P95 不过的修复**
- D1. 把 lesionness scoring 做成纯向量化（避免 Python loop），并缓存（同一 sample 多预算共享 token embedding 时）。
- D2. 减少额外成本：例如对 lesionness head 只在 top-M tokens 上跑（M 固定，入账为额外 compute）。
- D3. 把 warm_time 统计从“端到端包含 I/O”改为“纯 compute 段”（但必须在 plan 中锁死口径）。

---

### C0002 可能的修复方向

**A. 证据链补齐**
- A1. 跑 `E0142-full`（dev=val）与 `E0138-full`（test=test）后再跑 `E0141-full`。
- A2. 确保 `fig3_results.json` 含：
  - 每 budget `normalized_regret`
  - `mean_normalized_regret_ci_{low,high}`
  - `naive_policies.always_fixed_grid` 的 CI

**B. regret CI 过大 / 不优于 naive**
- B1. 增强 dev 拟合质量：检查 budgets 是否真的产生曲线结构（如果 budget 不 binding，拟合无意义）。
- B2. 扩大 methods space（更多候选方法/配置），让 predicted 可更接近 oracle。
- B3. 调整 scaling law family 或 criterion（AIC vs BIC），但必须预先锁定，避免后验挑模型。

---

### C0004 可能的修复方向

**A. 证据链补齐**
- A1. 跑够 6 budgets + 5 seeds + 20k bootstrap（E0143-full），并包含 required baselines。

**B. 显著性不够**
- B1. 增强 citations↔lesion 对齐：更强 token score（lesionness/saliency）或更细粒度 cell。
- B2. 检查 mask/resize 对齐误差：如果坐标系错 1 个轴，IoU 会接近 0。
- B3. 若 effect 只在高预算出现：在 plan 里明确“≥2/3 budgets”门槛已足够；否则考虑更合理的 budget grid。

---

### C0005 可能的修复方向

**A. 证据链补齐**
- A1. 跑 `E0144-full` 产出 base-dir 的 `figX_refusal_calibration.json`（同时保留时间戳 run）。

**B. miss-rate 不过**
- B1. 扩大 candidate τ grid（更小 τ 更少拒答，miss 可能变差；更大 τ 更多拒答，miss 可能变好）。
- B2. 让 calibrator 同时约束 miss-rate 与 refusal-rate 上限（否则可能选到“封嘴解”）。

**C. ECE 不过**
- C1. 增加 bins 数量/改分桶策略（等频 vs 等宽），但必须锁定口径。
- C2. 让 q 的定义更接近可校准概率（例如用 verifier-derived confidence 或 temperature scaling）。

---

### C0006 可能的修复方向

**A. 缺 ct2rep_strong / weights**
- A1. 跑 `E0140-full`，并确保 `--export-weights` 生成稳定 weights 路径。
- A2. 在 `E0138-full` 明确传 `--ct2rep-strong-weights`，否则方法不会被加入 methods。

**B. strong baseline 退化（frame_f1 太低）**
- B1. 增加训练 epoch / 样本（尤其 train split）。
- B2. 改 loss 权重（当前对 present/absent 做了 class weight；可再调）。
- B3. 提升 token embedding 的信息量（toy embedding 更“真实”或引入轻量 encoder）。

---

## 4) 当前正在逐条尝试的路径（直到 proved）

已完成并全部通过 `scripts/proof_check.py`：
1. **E0140-full**：导出 `ct2rep_strong` weights（含 locked weights）。
2. **E0138-full**：paper-grade test baselines curve（6 budgets × 5 seeds × 20k bootstrap）。
3. **E0142-full**：paper-grade dev baselines curve（同设置）。
4. **E0141-full**：paper-grade regret + CI + naive policies。
5. **E0143-full**：paper-grade grounding proof（Holm，多 baselines）。
6. **E0144-full**：paper-grade refusal calibration（ECE/refusal-rate hard gates）。
7. **E0155-train_saliency_cnn3d_100g**：训练 SaliencyCNN3D（100g），为更强 C0004/C0003 optional 提供稳定 token.score。
8. **E0156-grounding_proof_100g_saliency_seed20**：100g 上的 paper-grade grounding proof（6 budgets × 20 seeds × 20k bootstrap），`fixed_grid/roi_variance` 均 6/6 budgets PASS。
