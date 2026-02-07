# Oral Pitch (30s + 120s)

> 版本：2026-02-07（已同步 E0166/E0167 full）

## 30s 版本
我们做的是 ProveTok：在严格预算 `B=B_enc+B_gen` 下，把 3D 证据 token 化，并要求每条结论都携带可机检 citation 和 verifier trace。核心结论有三点：第一，在真实口径下我们已满足 multi-budget 的质量与 latency/trust gate；第二，counterfactual 显示去掉证据会显著伤害 grounding，不是“会说但没证据”；第三，跨数据集 v0003 已补齐：在 CT-RATE 外部 TS-Seg eval-only 子集上，grounding 对 ROI baseline 在 6/6 budgets 显著，且 `no_cite` 在 3/3 seeds 显著下降。我们也明确边界：该跨域证据属于 `silver_auto_unverified`，用于泛化趋势，不替代 gold-mask 主结论。

## 120s 版本
我们的目标不是把 3D 报告模型做得“看起来更聪明”，而是在可审计约束下证明它在预算、证据和可靠性上都成立。

第一层是预算与公平性。我们把问题写成联合预算 `B=B_enc+B_gen`，并在 FLOPs-matched 和 latency-aware 条件下比较方法，而不是只报一个点的平均值。当前主线证据显示，在 real profile 下关键 gate 已闭环，包括质量、tail latency 与 trust 约束。

第二层是 proof-carrying generation。ProveTok 不只输出文本，还输出可追溯的 citations、支持域与 verifier trace；当证据不足时触发 refusal，并受 critical miss-rate 约束，避免“封嘴换指标”。

第三层是“citations 不是装饰”。counterfactual 里我们系统地做 `omega_perm/no_cite/cite_swap/evidence_drop`。在 v0003(A') 的 CT-RATE TS-Seg eval-only 集合上，`no_cite` 在 3 个 seeds 都显著降低 grounding（Holm 后 `p=0.0`）；`omega_perm` 方向一致但未达显著，这给了我们一个诚实且可复核的结论：证据通路有效，但扰动强度与统计功效仍需在更强标注上继续验证。

最后是跨域叙事。我们已完成两条互补路径：
- V0003(C) 弱标签 pseudo-mask：证明跨域流程可跑通，并在多 seed 下观察到稳定方向；
- V0003(A') 外部自动分割来源（TS-Seg）：比自生成 pseudo-mask 更独立，进一步支持“跨域趋势存在”。

我们不会把这两条说成 gold-level 结论。当前口径是：它们显著降低了“只在单一数据集成立”的风险，但最终强证据仍需 A/B 路径，即真实 mask 或人工标注 eval 子集。

## 现场补充（一句话）
如果被问“你最薄弱的一环是什么”，标准回答是：跨域证据目前仍是 silver 级，我们已经用双路径把趋势做实，下一步是把同一协议迁移到 gold-mask 子集完成闭环。
