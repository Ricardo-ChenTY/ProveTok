# Section Templates (English) — NeurIPS Paper Draft for ProveTok

目标：把 `README.md` 的信息重排成“顶会论文可读”的英文段落模板。这里提供可直接套用的段落骨架与句式，不包含最终数值（数值必须引用 `docs/paper_assets/*` 与 `outputs/*`）。

提示：每段建议最多 5-7 句；每段必须只有一个“主目的”，并能被一条审稿追问打到要害时仍可防守。

---

## Abstract (6–8 sentences)

段落骨架（moves）：
1. Problem: In 3D CT report generation, the hard failure is not fluency but unsupported clinical claims under strict compute budgets.
2. Gap: Existing systems often treat grounding/citations as post-hoc explanations and refusal as a thresholding hack, making latency, supportedness, and safety hard to audit jointly.
3. Idea: We propose **ProveTok**, a proof-carrying generation protocol that couples **budgeted evidence tokenization (BET)** and **proof-carrying generation (PCG)** under a unified budget `B = B_enc + B_gen`.
4. Mechanism: BET produces evidence tokens with explicit 3D support regions `Ω`; PCG requires each clinical statement to carry machine-checkable citations and a verifier trace, otherwise it triggers refinement or calibrated refusal.
5. Audit: We define claim-level, scriptable proof rules and evaluate under multi-budget, multi-seed protocols with paired bootstrap and Holm correction.
6. Result highlight: Summarize 1–2 headline outcomes with pointers (e.g., “all primary claims pass under the real profile; pooled counterfactual tests show non-trivial citation effects”).
7. Implication: The key message: citations are not decorative; trust/refusal is not achieved by silencing.

---

## Introduction (3–5 paragraphs)

### P1: Why this problem, why now
- Start with the clinical risk: unsupported assertions are costly; 3D CT makes evidence localization harder.
- Tie to compute: 3D volumes are redundant; fixed-grid tokenization is expensive; budgeted inference is inevitable.

### P2: What is missing in prior work
- Prior 3D report generation: strong text metrics do not imply evidence.
- Prior grounding: often evaluated post-hoc, not enforced by the generation protocol.
- Prior refusal: can reduce apparent error by over-refusing; needs calibration with miss-rate constraints.

### P3: Our reframing (one paragraph)
- “We rewrite 3D report generation as a coupled budgeting + proof-carrying problem.”
- Define the output contract: `frames + citations + refusal + verifier_trace`.
- Define what is machine-checkable, not subjective.

### P4: Contributions (bullet list, 3 items)
Each bullet must be checkable and should end with an evidence pointer (Fig/Table):
- Protocol: BET + PCG + verifier taxonomy + calibrated refusal as one auditable loop (Fig1).
- Evaluation: multi-budget, matched-cost comparisons and claim-level proof rules (Table4).
- Evidence: counterfactual and calibration studies showing non-trivial citation effects without “silence for safety” (Fig4, Fig6).

---

## Related Work (2–4 paragraphs + one “positioning” paragraph)

Write related work as “gap closing,” not a bibliography dump:
- 3D CT report generation and datasets (CT-RATE/CT2Rep).
- Pixel-level grounding in CT (ReXGroundingCT).
- Trustworthy generation, attribution, and learning to refuse (Trust-Align).
- Positioning paragraph: “We are not proposing a new backbone; we propose a proof-carrying protocol with auditable gates under matched budgets.”

---

## Method (5–8 subsections, but keep each short)

Recommended flow:
1. Problem Setup: define `V`, `B`, outputs; define what counts as “supported”.
2. BET: evidence tokens, `Ω`, cost accounting (how `B_enc` is measured).
3. PCG: statement generation with mandatory citations; how verifier interacts.
4. Verifier taxonomy: what failure types exist; what actions are triggered.
5. Refusal calibration: define `τ_refuse`, ECE, and *critical miss-rate* hard gate.
6. Proof rules: claim-level pass/fail definitions (briefly), leaving full details to appendix.

Sentence patterns that help defend “engineering vs research” criticism:
- “Our contribution is the *contract* and *auditable gates* that make evidence and refusal machine-checkable under fixed budgets.”
- “Each module is simple; the novelty is the end-to-end verifiability and statistically guarded evaluation protocol.”

---

## Experiments & Results (structure that matches the figures/tables)

### Setup (one section)
Minimum must-include items:
- Datasets and profiles: clearly separate gold-mask (real) vs silver/pseudo-mask (stress tests).
- Baselines and matched-budget protocol (FLOPs/latency).
- Statistics: seeds, paired bootstrap, Holm, confidence intervals.

### Results (one section)
Recommended order (so it reads like an oral):
1. Main claim-level evidence (Table4).
2. Multi-budget curves and Pareto tradeoffs (Fig2).
3. Allocation model and regret (Fig3).
4. Counterfactual non-triviality (Fig4) + cross-dataset summary (Table2/appendix).
5. Refusal calibration with anti-silencing gate (Fig6).
6. Qualitative cases (Fig5).

---

## Discussion / Limitations

Two must-have blocks:
- Why the evidence is convincing: emphasize auditability + matched protocol + statistics + hard gates.
- Boundaries and failure modes: gold vs silver, when `omega_perm` needs power, where baselines may be incomplete.

Close with 2–3 future work items that are credible and aligned with the claims.

