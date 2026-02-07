# Draft Review Log — 5-pass Iteration (ProveTok NeurIPS LaTeX)

说明：本日志用于记录“逐段 review → 润色”的 5 轮迭代，目标是让正文达到顶会 oral 水平：可讲清、可防守、可复现、可追责。

规则（每轮都遵守）：
- 不改结论口径：所有数值只来自 `docs/paper_assets/*` 与 `outputs/*`。
- 每段必须满足：主目的单一 + 紧邻证据指针 + 可防守边界。

---

## Round 1 — Clarity (读者能否无痛跟上)
- Checklist:
  - 每段首句是否回答“这段唯一目的是什么”
  - 是否有概念未定义（`B_enc/B_gen/Ω/τ_refuse` 等）
  - 是否存在 README 式口语与跳跃

- Applied edits (2026-02-07):
  - `paper/sections/abstract.tex`: 明确 `Ω` 是 token 绑定的 3D 支持域（multi-resolution cell），避免抽象名词悬空。
  - `paper/sections/introduction.tex`: 显式定义 frame=原子临床断言，并把“携证据上岗”从口号改成契约描述（frame 必带 citations+verifier trace）。
  - `paper/sections/method.tex`: 补齐形式化定义（token/Ω、frame y_k、citations C_k、verifier g），并加入 PCG 的 3-step loop，降低读者认知跳跃。
  - `paper/sections/results.tex`: 在 Table 前增加一段“每条 claim 的一句话含义”，让 C0001..C0006/V0003 不再像内部编号。

## Round 2 — Defensibility (每句话能否被追问顶住)
- Checklist:
  - 每个 claim 是否紧贴 Fig/Table/JSON path
  - 是否存在“看起来像结论，但没有协议”的句子
  - 是否写清 matched/seed/bootstrap/Holm

- Applied edits (2026-02-07):
  - `paper/sections/method.tex`: refusal 口径落地到可审计定义：`q_k`=support probability（对齐 verifier-supported 而非 clinical correctness），critical miss-rate 定义与固定 critical set（pneumo/effusion/consolidation/nodule），并强调 dev 选阈值、test 冻结。
  - `paper/sections/experiments.tex`: metrics 逐条写成可追责定义（unsupported=verifier U1，critical miss-rate=GT critical 被拒答），统计协议补齐常用 `n_boot=20k`。

## Round 3 — Positioning (相关工作差异点是否明确)
- Checklist:
  - 是否明确“不是新 backbone，而是协议/审计闭环”
  - 与 CT2Rep / Trust-Align / ReXGroundingCT 的差异句是否出现且准确

- Applied edits (2026-02-07):
  - `paper/sections/related_work.tex`: 加强“差异句”密度：对 CT-RATE/CT2Rep/GenerateCT 明确本文是 contract+audit+matched-budget；对 ReXGroundingCT 明确从“可评测”推进到“协议强制”；对 Trust-Align 明确本文场景是 3D 空间证据+预算约束，refusal 必须与 grounding/latency/supportedness 联审。

## Round 4 — Statistics & Fairness (统计与公平比较是否无漏洞)
- Checklist:
  - 多重检验与 family 的定义是否一致
  - 置信区间/显著性是否与图表一致
  - 是否避免 cherry-pick（预算/seed/指标）

- Applied edits (2026-02-07):
  - `paper/sections/experiments.tex`: 明确 Holm 的 family 定义（budgets family vs counterfactual family），并把“dev 选阈值、test 冻结”写进 matched 协议，减少被质疑为 post-hoc 调参/挑阈值。

## Round 5 — Oral Readiness (口头答辩的可讲性)
- Checklist:
  - 每节是否可压成 2 句讲完
  - Fig1→Fig2→Fig3→Fig4→Fig6 的讲解链是否顺滑
  - Discussion 是否提前回答“omega_perm 为什么需要 pooled 功效”与“拒答是否封嘴”

- Applied edits (2026-02-07):
  - `paper/sections/discussion_limitations.tex`: 增加 “Two-sentence oral summary”，把整篇论文压缩成可口述的 2 句闭环，并显式绑定到 Table~\ref{tab:oral-minset} 的可复现证据。
