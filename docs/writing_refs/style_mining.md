# Style Mining Notes (Related Work) — for ProveTok Paper Writing

目的：不是复述论文内容，而是拆解“顶会论文每个部分怎么写、图表怎么做、审稿人想看什么证据”，把可复用的写作套路沉淀下来，供 `paper/` 的 LaTeX 正文使用。

约束：
- 不复制对标论文的大段原文；只抽象结构、论证顺序、常用表头/图形类型、caption 信息密度。
- 所有 ProveTok 的数值与结论必须来自本仓库可复现产物（见 `docs/paper_assets/*` 与 `outputs/*`）。

本文对标集合（与 `docs/paper_assets/references.md` 一致）：
- CT-RATE / CT-CLIP / CT-CHAT (arXiv:2403.17834, 2024)
- CT2Rep (arXiv:2403.06801, 2024)
- GenerateCT (arXiv:2305.16037, 2023)
- ReXGroundingCT (arXiv:2507.22030, 2025)
- Trust-Align (arXiv:2409.11242, 2024; ICLR 2025)

---

## 1) CT2Rep（3D report generation baseline）写作套路

### Abstract（常见 4-6 句 “moves”）
- 背景：radiology report generation 的临床价值与现状（2D 为主）。
- 缺口：3D 数据带来更强空间上下文，但训练/推理在计算与数据上更难，导致缺少强可复现基线与公平比较。
- 方法概览：一句话说清核心方案（例如“把 3D → text 的信息抽取改写成更可训练/更可控的表示学习流程”）。
- 贡献与产出：强调“提供 baseline + 数据处理 + 评测协议 + 代码”，并点到结果（不要堆数）。

### Introduction（典型结构）
- P1：先把“为什么重要”讲成审稿人同意的常识（医疗风险、效率、规模化）。
- P2：指出 3D 相比 2D 的 *unique pain*：体数据冗余、计算爆炸、标注/数据组织难、评测不统一。
- P3：给出“问题重写 + 方法一句话 + 贡献预告”，并明确“我们不是做花活结构，而是把 baseline 做强、把协议做清楚”。
- P4：贡献列表（一般 3 条），每条可被追责：数据/代码/基线/评测/扩展设置。

### Methods（写法重点）
- 先画系统图或流程图（通常在方法第一屏出现），让读者不读公式也能跟上。
- 再用 2-4 个小节写清：输入/输出定义、关键模块、训练目标/损失、推理流程（可含伪代码）。
- 细节（预处理、数据过滤、实现超参）往后放到 “Implementation Details” 或 appendix。

### Experiments（表格套路）
- 主结果表通常是 “Dataset × Metric” 的大表，带 baseline、加粗 best、必要时带 ±std 或 CI。
- 消融表：每行一个组件开关（w/o X），列是关键指标（质量 + 可信/grounding + 效率）。
- 定性图：常用多行样例，每行包含输入/输出/证据（图像 patch 或 attention heatmap）与短评。

### 可迁移到 ProveTok 的要点
- 用一张“主结果表/主证据表”尽早定调（ProveTok 对应 Table 4）。
- 把“公平比较”写死：FLOPs/latency matched、seeds、bootstrap/Holm 这些必须在 setup 里出现一次，后面只引用。

---

## 2) CT-RATE / CT-CLIP / CT-CHAT（multimodal foundation + dataset）写作套路

### 结构特征（更像 dataset/foundation paper）
- Intro 后面往往先进入 “Results” 再补 “Methods”（或把方法细节放在后半部分/appendix），强调：先给规模与任务，再给实现细节。

### Intro（典型 moves）
- P1：动机是 *scale* 与 *generalist*：更大更丰富的 CT 数据让 foundation 成为可能。
- P2：指出现有数据/任务碎片化，跨任务评测缺少统一协议。
- P3：贡献列表会包含：数据集构建、任务套件、基线/模型、开放资源。

### 图表套路
- Fig1 往往是 “dataset overview + tasks overview”的大图（多面板），caption 信息密度很高。
- 数据集统计表：包含 scans/reports/modalities/label types；会强调 train/val/test 划分与泄漏防护。

### 可迁移到 ProveTok 的要点
- “资源与协议”也是贡献：ProveTok 的 proof gate、资产生成脚本、可审计输出格式，应该写成可引用贡献，而不是 README 附录。

---

## 3) ReXGroundingCT（pixel-level grounding dataset）写作套路

### 关键词：定义先行 + 标注管线图 + 统计表
- 先把任务定义写死：sentence-level grounding、什么叫 finding、mask 的含义、评测口径（IoU/Dice/hit-rate）。
- 立刻给一张 “dataset construction pipeline” 的流程图，标注每步输入输出与质检点。
- 数据统计表与分布图紧跟着出现：finding 类型分布、mask 面积分布、slice/volume 维度。

### Limitations（写法）
- 很直接：覆盖不足/偏倚/泛化边界/标注误差来源；不会藏着。

### 可迁移到 ProveTok 的要点
- ProveTok 必须明确：哪些结论只在 gold-mask (real profile) 上成立，哪些是 silver 路径压力测试（避免“跨域银标当主结论”的致命误读）。

---

## 4) Trust-Align（grounded attribution + learning-to-refuse）写作套路

### Intro（叙事抓手：风险故事 → 现有范式 → 缺口 → 方法）
- 先用“可被共识的风险场景”建立紧迫性（hallucination/unsupported 的真实代价）。
- 再说 RAG/attribution 的趋势，但指出 attribution 质量与拒答行为需要一起被约束，否则会出现“封嘴刷安全”的问题。
- 方法一句话：定义一个可度量的 trust 指标或训练目标，并提出对齐/优化策略。
- 贡献：通常强调 metric + method + evaluation suite。

### Experiments（表与图）
- 主表把 “helpfulness/accuracy” 与 “trust/calibration/refusal” 放在一起（同一张表或同一组 figure），强迫读者相信不是 tradeoff 偷换概念。
- 常见图：reliability diagram、ECE 曲线、refusal-rate vs error/miss 的曲线；caption 会写清阈值冻结与评测条件。

### 可迁移到 ProveTok 的要点
- refusal 不能只给 ECE：要把 `critical_miss_rate` 写成 hard gate，并展示在多个 budget 上不会“封嘴”。

---

## 5) GenerateCT（3D text-conditional generation）写作套路

### 方法段落（更偏生成模型写法）
- Notation 很规范：随机变量/条件变量/目标分布/采样过程，定义一次后全篇复用。
- 实验会包含：训练细节（数据规模、步数、硬件）、质量度量（分布相似、下游任务、读片评估）、定性样例图。

### 可迁移到 ProveTok 的要点
- ProveTok 的“预算”是核心符号：在 LaTeX 里需要像生成模型那样统一符号体系（`B, B_enc, B_gen, Ω, τ_refuse`），避免 README 式口语化导致读者断线。

---

## 6) 顶会审稿默认期待的“最小图表集”（归纳）

强经验：读者/审稿人通常不愿意从文字里“自己拼结论”，所以需要 *一组* 固定图表骨架。

- Fig1：系统/协议概览（输入→BET→PCG→Verifier/Refusal→可审计输出），要能 30 秒讲完。
- Table (main)：主结果/主证据（ProveTok 对应 Table 4），越早出现越好。
- Fig2：核心 tradeoff 曲线（质量 vs 预算/延迟），体现 scaling 与 Pareto。
- Fig3：方法非平凡性/不是调参（allocation regret 或可解释模型拟合）。
- Fig4：反事实/因果压力测试（证明 citations/Ω 不是装饰）。
- Fig6：可信/拒答校准（要包含“防封嘴”的硬约束指标）。
- Fig5（可选）：定性案例，用于口头答辩与失败模式解释。

