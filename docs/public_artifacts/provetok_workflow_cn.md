# ProveTok 项目全流程中文图（闭环版）

> 目标：把当前项目从“计划→实现→实验→证明→回改计划”完整串成可持续迭代的工程闭环，并明确每一步在服务什么问题。

## 1) 全链路流程图

```mermaid
flowchart TB

classDef plan fill:#E8F1FF,stroke:#1D4ED8,stroke-width:1.5px,color:#0F172A;
classDef exp fill:#E8FFF3,stroke:#059669,stroke-width:1.5px,color:#0F172A;
classDef proof fill:#FFF3E8,stroke:#D97706,stroke-width:1.5px,color:#0F172A;
classDef gate fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,color:#0F172A;
classDef loop fill:#FEF3C7,stroke:#B45309,stroke-width:1.2px,color:#0F172A;
classDef end fill:#DCFCE7,stroke:#15803D,stroke-width:1.8px,color:#052E16;

Start((开始<br/>明确目标与口径))

subgraph L1["A. 计划与实现闭环（Plan ↔ Mohu）"]
direction TB
S1["S1 项目结构扫描<br/>识别代码入口、数据路径、依赖与docs状态"]:::plan
S2["S2 plan↔mohu 同步<br/>生成/刷新未实现点与模糊点清单"]:::plan
G1{"G1 mohu清单<br/>是否已清空？"}:::gate
S3["S3 mohu逐条解决<br/>实现/澄清 + 必要验证闸门"]:::plan
end

subgraph L2["B. 实验与证据闭环（Ledger → Smoke → Full）"]
direction TB
S4["S4 实验台账维护<br/>更新 E####、命令、指标、资源预算"]:::exp
S5["S5 数据资产准备<br/>构建/校验manifest，确保可复现"]:::exp
S6["S6 训练3D空间模型<br/>CNN/UNet/ResUNet/VNet等"]:::exp
S7["S7 外部gold本地验证<br/>verify_rexrank_manifest 对齐能力审计"]:::exp
S8["S8 核心实验运行<br/>基线/消融/LLM合同实验"]:::exp
G2{"G2 smoke<br/>是否全部通过？"}:::gate
S9["S9 失败归因与修复<br/>仅修当前阻塞项后重跑"]:::loop
S10["S10 full队列执行<br/>tmux + .rd_queue 日志/结果留痕"]:::exp
G3{"G3 full<br/>是否全部通过？"}:::gate
end

subgraph L3["C. 证明与论文闭环（Proof → Plan）"]
direction TB
S11["S11 结果回填<br/>docs/results.md + docs/experiment.md"]:::proof
S12["S12 claim可证性审计<br/>按plan逐条检查证据覆盖"]:::proof
G4{"G4 所有claim<br/>是否可被证据支撑？"}:::gate
S13["S13 更新plan（保留前后）<br/>回注未证实点并生成下一轮需求"]:::loop
S14["S14 对外产出<br/>口头版材料/提交包/复现实验清单"]:::proof
Done((闭环完成)):::end
end

Start --> S1
S1 --> S2 --> G1
G1 -- 否 --> S3 --> S2
G1 -- 是 --> S4

S4 --> S5 --> S6 --> S7 --> S8 --> G2
G2 -- 否 --> S9 --> S4
G2 -- 是 --> S10 --> G3
G3 -- 否 --> S9
G3 -- 是 --> S11

S11 --> S12 --> G4
G4 -- 否 --> S13 --> S1
G4 -- 是 --> S14 --> Done
```

## 2) 每一步在做什么（作用说明）

| 编号 | 步骤 | 作用 | 关键产物 | 通过标准 | 失败时怎么处理 |
|---|---|---|---|---|---|
| S1 | 项目结构扫描 | 建立“当前真实实现”的客观基线，避免只看文档假设 | 代码入口、数据/权重路径、依赖快照 | 关键入口和路径可定位 | 回到扫描，补齐缺失信息 |
| S2 | plan↔mohu 同步 | 把计划与实现差异显式化，形成可执行 backlog | `docs/mohu.md` 两类清单（未实现/模糊） | 清单条目可落地执行 | 回写清单并细化条目 |
| S3 | mohu逐条解决 | 强制“实现+验证闸门”逐项过关，防止债务滚雪球 | 代码改动 + 验证记录 | 当前条目验证通过 | 只修当前条目直到通过 |
| S4 | 实验台账维护 | 定义“跑什么、怎么跑、看什么、何时算过” | `docs/experiment.md`（E####） | 每个 claim 至少有实验覆盖 | 补命令、补指标、补资源估算 |
| S5 | 数据资产准备 | 保证训练/评测可复现、可审计、可回放 | manifest / index / splits / meta | 路径有效、ProtocolLock 通过 | 修 manifest 与分割策略后重建 |
| S6 | 训练3D空间模型 | 学到体素级定位能力，为 grounding 提供空间证据 | saliency 权重与训练日志 | val 指标非退化且稳定 | 调模型/损失/采样并重训 |
| S7 | 外部gold本地验证 | 检查模型是否真“看对位置”，不是只会文本匹配 | `verify_rexrank_manifest.json` | hit/dice 达到目标区间并可解释 | 分析瓶颈（模型/映射/数据）再迭代 |
| S8 | 核心实验运行 | 完成论文主结论所需的基线、消融、LLM相关证据 | 各实验输出 JSON/日志 | 关键实验可复现并产出指标 | 定位失败命令并最小修复 |
| G2 | smoke 闸门 | 低成本筛错，阻止坏配置进入 full | smoke 结果 | 所有 smoke 通过 | 回 S9 修复 |
| S10 | full 队列执行 | 在受控队列中完成重实验，保证留痕与可恢复 | `.rd_queue/logs` / `.rd_queue/results` | full 全通过 | 回 S9 修复后重跑 |
| S11 | 结果回填 | 把“跑过”变成“可读、可查、可引用”的证据面 | `docs/results.md` / `docs/experiment.md` | 结果与命令一一对应 | 补齐缺失结果与路径 |
| S12 | claim可证性审计 | 审核每个 claim 是否有足够证据支撑 | proof 审计报告 | claim-证据映射完整 | 标记未证实点并退回迭代 |
| S13 | 更新 plan（保留前后） | 把失败/不足显式纳入下一轮目标，防止叙事漂移 | `docs/plan.md` 前后版本 | 新一轮目标可执行可验证 | 回到 S1 重启闭环 |
| S14 | 对外产出 | 输出可用于 oral/复审/提交的标准化材料 | 口头版材料、提交包、公开结果 | 外部可复核 | 不足则回 S12/S13 |

## 3) 这张图解决的核心问题

1. **为什么一直在“循环”**：因为顶会证据标准要求“实现-实验-证明”连续闭环，不能只跑一次。  
2. **为什么要有 smoke/full 双闸门**：先低成本发现结构性错误，再把算力用于真正可用配置。  
3. **为什么要外部 gold 验证**：它是空间 grounding 的客观锚点，直接决定结论可信度。  
4. **为什么 proof 后还要改 plan**：审稿关注的是“claim 是否被证据支撑”，不是“实验是否很多”。  

