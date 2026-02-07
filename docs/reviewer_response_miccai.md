# MICCAI/MIA Reviewer Response Notes (v0003)

本文档用于把你提供的合并 review（W1–W7）逐条落到“论文改哪、补哪张表图、需要跑哪些最小实验、对应代码/产物在哪里”，避免改稿变成口号。

> 约定：本文的 “proof” 指 **verifier-checkable evidence chain**（`citations + verifier_trace`），不是临床事实真值证明。

## W1. 方法细节不足（BET/PCG/Ω/token/verifier）

### Reviewer concern
- BET 的 scoring/selection 机制、Ω 与 token 表示、PCG 如何产出 `frames+citation`、verifier 如何判 unsupported 均描述不足，导致贡献看起来更像“概念/协议”而非可复刻机制。

### What we add (paper)
- `paper/sections/method.tex`
  - **Ω/Token 明确定义**：token 绑定到 3D support region `Ω_i`，在实现中为一个 dyadic cell（八叉树式 split）的体素 slice。
  - **BET refine loop 的算法描述**：从 coarse full-cover grid 或 `root_cell()` 起，迭代 `encode → PCG decode → verify → pick/split → stop`，并给出 stop 条件（budget reached / no issues / require_full_budget）。
  - **Allocator 与 EvidenceHead 的角色**：allocator 优先 split verifier trace 指向的 cell，否则 split 最大不确定性 cell；EvidenceHead 预测 `Δ(c)` 用于更稳定的 split ranking。
  - **PCG 结构化 head**：finding queries + citation attention → slot classifiers → frames；`q_k` accept prob head；refusal policy 的冻结阈值与 hard gates。
- `paper/sections/appendix.tex`（新增/扩展）
  - **Verifier ruleset 表**：列出 U1/O1/M1/I1 的判定输入、阈值、输出 trace 字段（`token_ids/token_cell_ids/rule_inputs/rule_outputs`），并写明 `RULE_SET_VERSION/TAXONOMY_VERSION`。

### Source of truth (code pointers)
- Token/Ω：
  - `provetok/types.py` (`Token`, `Generation`, `Issue`)
  - `provetok/grid/cells.py`（`Cell`、`cell_bounds` 给出 `Ω=phi(cell)`）
- BET：
  - `provetok/bet/tokenize.py`（`TokenEncoder`：encoder pooling 或稳定 toy embedding；cache）
  - `provetok/bet/refine_loop.py`（完整 refine loop、stop 条件、score 融合）
  - `provetok/bet/allocator.py`（issue-trace 驱动的 greedy split）
- PCG：
  - `provetok/models/pcg_head.py`（attention citations + slot heads + `q_k`）
  - `provetok/models/system.py`（evidence graph → constrained vocab；verifier-in-the-loop）
- Verifier：
  - `provetok/verifier/rules.py`（rule-based verifier，trace schema）
  - `provetok/verifier/taxonomy.py`（critical finding set；版本号）

## W3. supportedness 不是 clinical correctness（必要非充分）

### Reviewer concern
- 证据支持不等于医学结论正确；需要最小的 correctness 证据，避免仅证明“citation 改变行为”。

### What we add (paper)
- `paper/sections/experiments.tex`
  - 新增 “Clinical correctness proxies” 小节：报告 `frame_f1` 之外，增加 **critical findings 子集**的 `F1/recall`（pneumothorax/effusion/consolidation/nodule）。
  - 明确：这些是结构化 findings 的 correctness proxy，不替代医生评审。
- `paper/sections/results.tex`
  - 增加一个小表/补表：在 matched budget 下，supportedness/grounding 提升同时，不以 critical recall 明显下降为代价（CI 报告）。

### Implementation & artifact plan
- 已落地：
  - 代码已支持新增指标：`provetok/experiments/run_baselines.py`（写入新增 metrics），`provetok/experiments/baselines_curve_multiseed.py`（聚合与 CI/paired bootstrap 对齐，NaN-aware）。
  - 论文级 artifact：`outputs/E0171-full3/baselines_curve_multiseed.json`（含 `critical_present_f1`、`critical_present_recall`）。
  - 图：`docs/paper_assets/figures/fig8_critical_recall.png`（在 `paper/sections/results.tex` 作为 Fig.8 引用）。

## W4. “proof-carrying”措辞可能过强

### Reviewer concern
- “proof” 容易被理解为“证明医学事实正确”，而当前更多是“可机检的证据链/审计 trace”。

### What we change (paper)
- 在 Abstract/Intro/Limitations 的第一次出现处明确限定：
  - proof = verifier-checkable evidence chain（citations + trace），不等价 clinical truth。
- 把一些口号式句子改为边界更清晰的表述：supportedness 是必要条件；clinical correctness 需额外验证。

## W6. omega_perm 需要 pooled seeds 才显著（功效/稳健性）

### Reviewer concern
- seed 0..2 Holm 不显著，pooled 0..19 才显著，担心 effect 小/方差大/统计依赖性。

### What we add (paper)
- `paper/sections/results.tex`
  - 增加 per-seed 稳健性描述：报告 `positive_seed_count/seed_count`，并给出 seed-level mean_diff 分布（表或图）。
  - 明确一侧假设与 Holm family 定义，解释 seed 0..2 不显著属于功效不足而非方向不一致。

### Artifacts
- 现有 pooled 产物：`outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json`
- 已生成 seed-level 稳健性图：`docs/paper_assets/figures/fig7_omega_seed_stability.png`（源：同上 JSON）。

## W7. gold-mask 规模与外部有效性

### Reviewer concern
- 主证据依赖 gold masks 数据规模可能偏小；silver label 外部有效性要明确边界。

### What we change (paper)
- `paper/tables/datasets_profiles.tex`（或同名表格文件）
  - 把 real profile 的样本量、split、mask 来源写清楚。
  - V0003/TS-Seg 明确标注为 silver label stress test，不用于主结论。
- `paper/sections/experiments.tex` + `paper/sections/discussion_limitations.tex`
  - 在正文重复一次 “主结论以 gold-mask real profile 为准；silver 只做 stress test”。

## Pending experiments
- **E0165-full**：C0004 grounding proof 扩 seeds（0..19）强化统计稳定性（对应你提出的“全量 + 多 seed”）。
- **E0171-full**：新增 critical correctness proxy 指标并复跑主曲线（回应 W3）。

## Definition hygiene (for Q&A)
- supportedness：verifier issue `U1_unsupported` 的 frame-level 率（paper 与 `provetok/verifier/rules.py` 对齐）。
- critical miss-rate：refusal 评估中 “GT critical finding 被错误拒答” 的比率（`provetok/pcg/refusal.py`）。
- `RULE_SET_VERSION/TAXONOMY_VERSION`：写入 artifact meta，确保审计一致性。
