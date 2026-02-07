# Figure/Table Blueprints — Mapping ProveTok Assets to a NeurIPS Paper

目标：把仓库现有 `docs/paper_assets/figures/*` 与 `docs/paper_assets/tables/*` 映射到“顶会论文标准位置”，并规定 caption 与正文引用时必须携带的关键信息（避免读者看不懂、也避免审稿人抓住口径漏洞）。

---

## Figures (existing)

### Fig. 1 — System / Protocol Overview
- File: `docs/paper_assets/figures/fig1_system_overview.png`
- Where: end of Introduction or beginning of Method
- Caption must include:
  - `B = B_enc + B_gen` 的预算分解
  - BET 产生 evidence tokens 与显式支持域 `Ω`
  - PCG 输出 `frames + citations + refusal + verifier_trace`
  - Verifier taxonomy 与 refusal calibration 的位置（硬 gate）
- Text hook: “ProveTok is a *contract*: every statement must carry machine-checkable evidence under a fixed budget.”

### Fig. 2 — Multi-budget Curves (quality/grounding/latency)
- File: `docs/paper_assets/figures/fig2_budget_curves.png`
- Where: Results (early)
- Caption must include:
  - budgets 范围与数量（6 budgets）
  - matched protocol（FLOPs/latency matched）
  - seeds 数量（来自 `outputs/E0164-full/baselines_curve_multiseed.json`）
  - 主要指标定义（combined / IoU / latency）
- Text hook: “We show Pareto tradeoffs under matched budgets rather than unconstrained scaling.”

### Fig. 3 — Allocation Model / Regret
- File: `docs/paper_assets/figures/fig3_regret_sweep.png`
- Where: Results, after Fig2
- Caption must include:
  - regret 的定义（与 naive 对比）
  - bootstrap/CI 的统计口径（引用 setup）
  - 数据源 `outputs/E0161-full/fig3_regret_sweep.json`
- Text hook: “Budget allocation is non-trivial and predictable; we quantify regret to avoid post-hoc cherry-picking.”

### Fig. 4 — Counterfactual Power / Pooled Significance (omega_perm)
- File: `docs/paper_assets/figures/fig4_counterfactual_power.png`
- Where: Results, after Fig3
- Caption must include:
  - counterfactual family（no_cite / cite_swap / omega_perm 等）
  - pooled seeds 的范围（0..19）
  - one-sided test + CI + Holm（secondary family control）
  - evidence path `outputs/E0167R2-.../omega_perm_power_report.json`
- Text hook: “Citations are not decorative: counterfactual perturbations yield measurable, statistically guarded effects.”

### Fig. 5 — Qualitative Case Studies
- File: `docs/paper_assets/figures/fig5_case_studies.png`
- Where: end of Results or beginning of Discussion
- Caption must include:
  - 每个案例的三件套：finding text、citation tokens（或其可视化）、verifier outcome
  - 失败模式（若包含）要标注属于哪类 verifier failure
- Text hook: “Qualitative cases complement the proof rules: they explain *why* gates pass/fail.”

### Fig. 6 — Refusal Calibration (anti-silencing)
- File: `docs/paper_assets/figures/fig6_refusal_calibration.png`
- Where: Results (trustworthiness subsection)
- Caption must include:
  - `τ_refuse` 的冻结方式（dev→test）
  - 指标：refusal ECE / refusal rate / critical miss-rate（硬 gate）
  - “anti-silencing”: supportedness/unsupported 是否改善
  - evidence path `outputs/E0144-full/figX_refusal_calibration.json`
- Text hook: “Refusal is constrained: we calibrate it without increasing critical misses.”

---

## Tables (existing)

### Table 4 — Oral Minimal Evidence Set (Paper-Grade)
- File: `docs/paper_assets/tables/table4_oral_minset.md`
- Where: early Results (often the first table)
- What it buys: “主结论可审计”，把审稿人从“你讲得像”拉到“我能复现/追责”。
- LaTeX conversion requirements:
  - 列固定：Item / Verdict / Key Numbers / Protocol / Evidence
  - Evidence 列保留路径（短路径或脚注）
  - Protocol 里的统计口径要与 Experiments Setup 完全一致（避免双口径）

### Table 1 — Claim-level Machine Verdict (real profile)
- File: `docs/paper_assets/tables/table1_claims_real.md`
- Where: appendix or main (depending on page limit)
- Role: 细化 Table4 的 claim 细节；用于审稿追问时快速定位 proof rule。

### Table 2 — V0003 Cross-Dataset Summary
- File: `docs/paper_assets/tables/table2_v0003_cross_dataset.md`
- Where: appendix (recommended)
- Role: cross-dataset/weak-label 的边界条件，避免被误解为主结论。

### Table 3 — Omega Variant Search
- File: `docs/paper_assets/tables/table3_omega_variant_search.md`
- Where: appendix (recommended)
- Role: 证明不是“挑一个 omega 配置凑显著”，而是探索后锁定 baseline，再扩 seeds。

### Table 5 — V0004 Backbone Transfer (Summary)
- File: `docs/paper_assets/tables/table5_backbone_transfer.md`
- Where: External Validity subsection (recommended appendix;正文仅保留 1–2 句话结论 + 指向表格)
- Caption must include:
  - 对比 backbones 名称与实现（ToyPCG / Llama2PCG 等）
  - 冻结项：预算 sweep、verifier rules、refusal calibration（强调“无 per-backbone 重新调参”）
  - 统计协议：paired bootstrap + Holm(budgets)（每个指标一个 family）
  - 明确“positive mean\_diff 表示改进”的方向约定（尤其是 unsupported 指标）
- Text hook: “The contract is (partially) backbone-agnostic under a frozen protocol; we report where transfer breaks and why.”

---

## Add-if-needed (paper-grade but currently missing)

若页数允许，顶会常见会补 1–2 张表增强“可读性”：
- Dataset summary table（每个 profile：mask 类型、标注级别、用途：train/eval/stress）
- Cost accounting table（FLOPs/latency/VRAM，解释 matched 协议如何执行）

这些可以先放 appendix，避免正文被表格淹没。
