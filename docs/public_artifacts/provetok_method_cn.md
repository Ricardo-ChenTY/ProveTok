# ProveTok 项目方法流程图（中文版，按代码实现）

> 这份图只描述“方法本体”如何工作（训练 + 推理 + 验证 + 评测），不描述研发管理闭环。

## 1) 方法总图

```mermaid
flowchart TB

classDef data fill:#E0F2FE,stroke:#0369A1,stroke-width:1.4px,color:#0F172A;
classDef bet fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.4px,color:#0F172A;
classDef gen fill:#ECFDF5,stroke:#047857,stroke-width:1.4px,color:#0F172A;
classDef verify fill:#FEF3C7,stroke:#B45309,stroke-width:1.4px,color:#0F172A;
classDef train fill:#FCE7F3,stroke:#BE185D,stroke-width:1.4px,color:#0F172A;
classDef gate fill:#F8FAFC,stroke:#334155,stroke-width:1.6px,color:#0F172A;
classDef artifact fill:#FFF7ED,stroke:#C2410C,stroke-width:1.4px,color:#0F172A;
classDef note fill:#F1F5F9,stroke:#475569,stroke-width:1.2px,color:#0F172A;
classDef terminal fill:#DCFCE7,stroke:#15803D,stroke-width:1.8px,color:#052E16;

Start((输入：3D CT体数据 + 预算B<br/>B = B_enc + B_gen)):::data

subgraph Infer["A. 在线推理主链（方法本体）"]
direction TB
I1["I1 BET初始化<br/>root cell / fixed-grid"]:::bet
I2["I2 TokenEncoder编码<br/>embedding + score + uncertainty"]:::bet
I3["I3 可选评分融合<br/>lesionness / saliency"]:::bet
I4["I4 EvidenceHead估计Δ(c)<br/>issue_reduction + uncertainty"]:::bet
I5{"I5 停机条件？<br/>预算到顶 / Δ<ε / 无可分cell"}:::gate
I6["I6 split c*并重编码<br/>细化高价值空间区域"]:::bet
I7["I7 EvidenceGraph构图<br/>生成constrained vocab"]:::gen
I8["I8 PCG解码<br/>frames + citations + q + refusal"]:::gen
I9["I9 Verifier规则校验<br/>U1/O1/I1/M1 + trace"]:::verify
I10{"I10 仍有高严重issue且可继续？"}:::gate
I11["I11 回到refine<br/>继续预算分配"]:::bet
I12["I12 输出双通道结果<br/>结构化frames + narrative文本"]:::artifact
I13["I13 评测与审计<br/>frame_f1 / iou_union / unsupported / counterfactual"]:::artifact
end

subgraph Train["B. 训练主链（M0→M3）"]
direction TB
T1["T1 数据与监督构建<br/>FrameExtractor + lesion masks"]:::data
T2["T2 M0/M1<br/>slot CE + citation弱监督"]:::train
T3["T3 M2<br/>+ grounding consistency loss"]:::train
T4["T4 M3<br/>+ LLM contract/refusal约束"]:::train
T5["T5 参数更新<br/>PCGHead + EvidenceHead (+可选LLM)"]:::train
end

Start --> I1 --> I2 --> I3 --> I4 --> I5
I5 -- 否 --> I6 --> I2
I5 -- 是 --> I7 --> I8 --> I9 --> I10
I10 -- 是 --> I11 --> I4
I10 -- 否 --> I12 --> I13 --> Done((方法一次推理完成)):::terminal

T1 --> T2 --> T3 --> T4 --> T5
T5 -. 训练得到的参数与策略 .-> I8

subgraph Notes["旁注节点（回答常见疑问）"]
direction LR
N3D["3D模型：I1-I3<br/>把体素变成token并打分，决定空间证据。"]:::note
NLLM["LLM：I8 + T4<br/>推理端生成frame/text；训练端学合同与拒答。"]:::note
NRef["Refine：I2-I4循环<br/>用Δ(c)把预算投到高价值区域。"]:::note
NVer["Verifier：I9<br/>抓unsupported/overclaim并决定是否返工。"]:::note
NOut["输出：I12-I13<br/>文本只是结果之一，必须有trace与评测证据。"]:::note
N3D --> NLLM --> NRef --> NVer --> NOut
end

Done -.-> N3D
```

## 2) 每一步作用（对应代码）

| 编号 | 作用 | 关键输入 | 关键输出 | 代码锚点 |
|---|---|---|---|---|
| I1 | 初始化 BET 空间划分（cell 集） | 3D volume, `B_enc` | 初始 cell 集 | `provetok/bet/refine_loop.py` |
| I2 | 把 cell 编码成 token（含 `cell_id/score/uncertainty/embedding`） | cell 集、volume、encoder | token 列表 | `provetok/bet/tokenize.py` |
| I3 | 融合 lesionness / saliency 分数，提升引用相关性 | token embeddings、评分头 | 重打分 token | `provetok/experiments/run_baselines.py` |
| I4 | 估计每个 cell 的边际收益 `Δ(c)` | 当前 issues + token embedding | 可 split cell 排序 | `provetok/bet/evidence_head.py` |
| I5-I6 | 在预算与收益约束下 split 最优 cell 并循环 | `Δ(c)`、`max_depth`、`epsilon` | 细粒度 token 覆盖 | `provetok/bet/refine_loop.py` |
| I7 | 从 token 构建证据图并导出 constrained vocab | token 列表 | `V_slot` 合法域 | `provetok/pcg/evidence_graph.py` |
| I8 | 生成 proof-carrying 输出（frames/citations/q/refusal） | token + `V_slot` | `Generation` | `provetok/models/pcg_head.py`、`provetok/pcg/generator.py`、`provetok/pcg/llama2_pcg.py` |
| I9 | 用规则 taxonomy 检测 unsupported/overclaim 等 | generation + tokens | issue 列表 + trace | `provetok/verifier/rules.py` |
| I10-I11 | 高风险 issue 触发继续 refine；否则停机 | issue 严重度 + 预算剩余 | 继续细化或输出 | `provetok/bet/refine_loop.py` |
| I12 | 输出双通道结果（结构化 + 可读文本） | frames/citations/refusal | narrative + trace | `provetok/types.py`、`provetok/pcg/narrative.py` |
| I13 | 量化方法有效性与可信性 | generation + GT/mask | frame/grounding/trust/cf 指标 | `provetok/eval/metrics_frames.py`、`provetok/eval/metrics_grounding.py`、`provetok/eval/counterfactual.py` |
| T1-T5 | M0→M3 训练路线（逐步加入 grounding 与 LLM） | stage config + dataset | 训练权重与策略 | `provetok/training/stages.py`、`provetok/training/trainer.py` |

## 3) 这张图如何解答“方法到底做了什么”

1. **预算如何生效**：I1~I6 明确 `B_enc` 用在“哪些空间区域被 token 化”。  
2. **证据如何绑定到文字**：I7~I8 把 token 证据强绑定到每个 frame 的 citation。  
3. **可信性怎么保证**：I9~I11 通过 verifier+refine 循环抑制 unsupported/overclaim。  
4. **为什么可审计**：I12~I13 输出结构化 trace 并可复算到 grounding/counterfactual 指标。  

## 4) 你最关心的两个问题（直答）

1. **我们为什么需要 3D 模型？**  
   因为任务要做的是“3D 空间证据绑定”，不是纯文本生成。3D 模型在 `I2/I3` 把体素区域转成 token 并打分，直接决定 citations 指向哪里；没有这一步，后面的 grounding 指标会塌。

2. **LLM 在哪里，负责什么？**  
   LLM 在 `I8`（推理时生成 frame+文本）和 `T4`（训练时合同/拒答约束）。不启用 LLM 时，系统仍可由 `PCGHead/ToyPCG` 跑通结构化输出与验证链路，但文本表达能力和合同泛化通常更弱。  

## 5) 审稿人追问版（挂图下可直接答）

1. **为什么 `+0.001` 的提升也可能有意义？**  
   前提是它在多预算、多 seed、`Holm` 校正后都稳定同向，且不牺牲 `unsupported`、`critical_miss_rate`、`refusal_ece`。这种结论应表述为“稳定小效应（protocol-level）”，而不是直接宣称“临床显著收益”。

2. **什么时候不该 claim 有效？**  
   只要出现以下任一条，就不该写成“有效”：`CI` 跨 0；`Holm` 后不显著；只在 silver 数据成立而 gold 证据缺失；或安全约束（`unsupported/miss-rate/refusal`）变差。

3. **oral 口径怎么避免被 reviewer 反打？**  
   用“边界清晰”的说法：  
   - 主结论：协议让证据绑定与审计更稳定；  
   - 次结论：效应量偏小但一致；  
   - 限制：仍需更强外部 gold 和临床端点验证。  
