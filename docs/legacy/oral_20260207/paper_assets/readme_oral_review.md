# README Oral-Level Review Log (5 Rounds)

目标：把 `README.md` 的论文式叙事打磨到“CS AI 顶会 oral 可防守”的标准：每个结论可追溯到产物路径、每个强断言都有统计口径/边界、每节都能在 Q&A 里用 1 句话回答“你到底证明了什么”。

本日志记录 5 轮迭代的逐段 review 与改写要点；最终文本以 `README.md` 为准。

## 段落单元（Unit）
以下以“空行分隔”的 Markdown 块为单位（U01..），用于逐段点评与追踪改动。

- U01 `# Title`
- U02 顶部状态栏（审计状态/强证据路径）
- U03 Abstract（含 1 段）
- U04 TL;DR（含 3 bullets）
- U05 Introduction 1.1（1 段）
- U06 Introduction 1.2（目标列表）
- U07 Introduction 1.3（贡献列表）
- U08 Related Work 2.1（1 段）
- U09 Related Work 2.2（1 段）
- U10 Related Work 2.3（1 段）
- U11 Related Work 2.4（1 段）
- U12 Method 3.1（问题定义+预算分解）
- U13 Method 3.1（目标句）
- U14 Method 3.2（BET 列表）
- U15 Method 3.3（PCG 列表）
- U16 Method 3.3（强约束句）
- U17 Method 3.4（Verifier/Refusal 列表）
- U18 Method 3.5（metrics 定义）
- U19 Method 3.6（Fig1）
- U20 Fig1 图注（证据路径）
- U21 Experimental Setup 4.1（datasets 表）
- U22 Setup 4.2（methods 列表）
- U23 Setup 4.3（protocol 列表）
- U24 Results 5.0（Table4 + 说明）
- U25 Results（Table4 生成文件指针）
- U26 Results 5.1（Fig2）
- U27 Fig2 图注（证据路径）
- U28 Results 5.2（Fig3）
- U29 Fig3 图注（证据路径）
- U30 Results 5.3（Fig4）
- U31 Fig4 图注（证据路径）
- U32 Results 5.4（Fig6）
- U33 Fig6 图注（证据路径）
- U34 Results 5.5（Table2）
- U35 Table2 指针
- U36 Results 5.6（Table3）
- U37 Table3 指针
- U38 Results 5.7（Fig5）
- U39 Fig5 图注（证据路径）
- U40 Discussion 6.1（bullets）
- U41 Discussion 6.2（边界）
- U42 Discussion 6.3（下一步）
- U43 Conclusion（1 段）
- U44 Reproducibility（脚本）
- U45 Proof gate（命令）
- U46 References（条目）
- U47 References 附注（references.md/.bib）

## Round 1：证据与口径收紧（可检索、可复现、可追责）
改写目标：删除/弱化无法被产物直接支撑的表述；把关键命令与口径写成“不会误导读者”的最短形式。

- U04：TL;DR 的审计命令补齐为可直接运行的严格版本（`oral_audit.py --sync --out ... --strict`）。
- U12–U13：补充输出定义（`frames + citations + refusal + verifier_trace`），避免“只说优化目标不说输出物”。
- U46：去掉未校验的 venue 断言（Trust-Align 仅保留 arXiv 信息）。
- `docs/paper_assets/references.md/.bib`：同步去掉未校验的“oral/venue”宣称，保持引用可核验。

## Round 2：叙事与贡献强化（oral 讲得清：问题-方法-证据-边界）
改写目标：把 Abstract/Intro 的信息密度拉满但不堆砌；把贡献写成可判定的 claims（对应 Table4）。

- U03（Abstract）review：
  - 问题：原文信息密度高但句式偏长，且“为什么可信”的协议与“证据在哪里”没有在同一节奏里讲清。
  - 改写：按“问题(预算+可信)→方法(闭环协议)→协议(多预算/多seed/统计)→证据(Table4/V0003)”四步重排，并显式写出预算范围 `2e6..7e6` 与输出物定义。
- U05（1.1）review：
  - 问题：与 Abstract 重复，缺少“为什么现有方法不够”的机制性解释。
  - 改写：把失败模式具体化为“无证据结论/过度断言/后处理拒答”，并指出预算约束下 grounding 是资源分配问题。
- U06（1.2）review：
  - 问题：目标较抽象，oral 现场不易把目标与 gate/实验对齐。
  - 改写：把目标写成可判定的口径（固定预算、多预算范围、输出携证、refusal hard gate、counterfactual+cross-domain）。
- U07（1.3）review：
  - 问题：贡献点缺少“如何被证明”的落脚。
  - 改写：用“协议/裁判/证据”三条贡献，每条都指向可复现脚本或 Table4。

## Round 3：Related Work 精修（不堆文献，强调“本文差异点”）
改写目标：每个 related work 小节末尾必须有 1 句“我们缺口在哪里/我们怎么补”，并避免泛泛而谈。

- U08–U11 review：
  - 问题：原文每小节仅 1 句总结，缺少“本文差异点”落脚，oral 现场容易被追问“那你和这些工作到底差在哪”。
  - 改写策略：每节追加 1 句（或并入同段）把差异写死，并尽量指向可判定证据（Table4 的 C0004/C0005）。
- U08（2.1）改写：强调 CT-RATE/CT2Rep 的价值在数据与强基线，但缺少可审计硬约束；明确 ProveTok 是协议与裁判而非 backbone。
- U09（2.2）改写：把 ReXGroundingCT 的意义落到“citation→空间覆盖→IoU_union”的可检验链，并指向 C0004。
- U10（2.3）改写：把 refusal 从“一个安全点子”落到“hard gate + 联动指标”，并指向 C0005。

## Round 4：Method/Setup 口径对齐（把 Q&A 可能追问的定义提前写死）
改写目标：把 metrics、统计检验、silver/gold 边界写到读者看得到的位置；避免 reviewer 抓“定义没说清”。

- U21（Datasets）review：
  - 问题：V0003(A'/C) 的 silver label 边界只在 Discussion 出现，容易被误解为主结论证据。
  - 改写：在数据集表后追加一句“silver 仅用于 sanity/压力测试；主结论以 ReXGroundingCT gold mask 为准”。
- U22（Methods）review：
  - 问题：README 写 `ct2rep_noproof`，但关键产物（E0164）里 method key 为 `ct2rep_like`，口径可能让读者困惑。
  - 改写：以产物 key 为准写 `ct2rep_like`，并声明若复跑产物名为 `ct2rep_noproof` 视为同义；同时更新图生成脚本兼容两种命名。
- U23（Protocol）review：
  - 问题：统计口径过粗（只写 bootstrap+Holm），缺少方向性假设的说明。
  - 改写：补一句“方向性假设用 one-sided；跨预算/反事实 family 用 Holm”并保持不夸大。

## Round 5：语言与排版终审（每句话都可防守）
改写目标：删冗余、补过渡、统一术语（IoU_union/unsupported_rate/critical_miss_rate）、保证 figure/table 引用一致。

- U12–U18 review：
  - 问题：Method 段落出现中英混用的抽象短语（例如 trust/latency hard gate），口径可更直接。
  - 改写：将关键句改为“可信性与延迟的 hard gates”，并保持术语不漂移。
- U32–U33（Refusal）review：
  - 问题：图注只说“满足 gate”，缺少“满足 gate 的同时改善了什么”的一句话结论。
  - 改写：补一句“冻结阈值并满足 gate，同时 6/6 budgets 降低 unsupported_rate”，并指向 C0005。
- U40（Discussion）review：
  - 问题：原 bullets 偏口号式（“不是…而是…”），oral 现场不如直接指证据与规则。
  - 改写：把每条 bullet 直接落到 Table4 的 claim（C0001/C0004/C0005/C0003/V0003），让“可信”可被逐条追问。
- U43（Conclusion）review：
  - 问题：需要更醒目地区分 gold-mask 主结论与 silver-label 压力测试结论。
  - 改写：明确 `real` profile=gold mask、V0003=silver label，并保持结论措辞保守可防守。
