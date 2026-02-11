# ProveTok 方法流程图（详细版：LLM作用位展开）

> 这份是“看清 LLM 在哪里起作用”的专用图。建议先看简版，再看这里。

## 1) 详细流程图

```mermaid
flowchart TB

classDef data fill:#E0F2FE,stroke:#0369A1,stroke-width:1.4px,color:#0F172A;
classDef bet fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.4px,color:#0F172A;
classDef gen fill:#ECFDF5,stroke:#047857,stroke-width:1.4px,color:#0F172A;
classDef llm fill:#DCFCE7,stroke:#15803D,stroke-width:1.6px,color:#0F172A;
classDef verify fill:#FEF3C7,stroke:#B45309,stroke-width:1.4px,color:#0F172A;
classDef train fill:#FCE7F3,stroke:#BE185D,stroke-width:1.4px,color:#0F172A;
classDef gate fill:#F8FAFC,stroke:#334155,stroke-width:1.6px,color:#0F172A;
classDef artifact fill:#FFF7ED,stroke:#C2410C,stroke-width:1.4px,color:#0F172A;
classDef note fill:#F1F5F9,stroke:#475569,stroke-width:1.2px,color:#0F172A;
classDef terminal fill:#DCFCE7,stroke:#15803D,stroke-width:1.8px,color:#052E16;

S0((输入：3D CT volume + 预算B<br/>B = B_enc + B_gen)):::data

subgraph BET["A. BET证据构建（3D空间部分）"]
direction TB
B1["B1 初始cells<br/>root/fixed-grid"]:::bet
B2["B2 TokenEncoder<br/>cell→(embedding,score,uncertainty)"]:::bet
B3["B3 可选打分融合<br/>lesionness/saliency"]:::bet
B4["B4 EvidenceHead估计Δ(c)<br/>issue_reduction + uncertainty"]:::bet
BG{"B5 停机？<br/>预算到顶 / Δ<ε / 无可分cell"}:::gate
B6["B6 split c* 并重编码"]:::bet
end

subgraph GEN["B. 生成与验证（文本/结构化部分）"]
direction TB
G0["G0 EvidenceGraph<br/>token→slot支持图 + constrained vocab"]:::gen
GG{"G1 生成后端选择<br/>pcg_backend = toy / llama2"}:::gate
G1["G2A 非LLM路径<br/>PCGHead/ToyPCG 解码"]:::gen
G2["G2B LLM路径<br/>Llama2PCG + contract_mode"]:::llm
G3["G3 统一输出<br/>frames + citations + q + refusal"]:::gen
V1["G4 Verifier规则集<br/>U1/O1/I1/M1 + trace"]:::verify
VG{"G5 高严重issue且可继续？"}:::gate
G4["G6 输出与评测<br/>text + trace + grounding/cf"]:::artifact
end

subgraph TRN["C. 训练阶段（M0→M3）"]
direction TB
T1["T1 M0/M1<br/>slot CE + citation弱监督"]:::train
T2["T2 M2<br/>+ grounding consistency loss"]:::train
T3["T3 M3<br/>+ LLM合同/拒答约束"]:::train
T4["T4 参数更新<br/>PCGHead + EvidenceHead (+可选LLM)"]:::train
end

S0 --> B1 --> B2 --> B3 --> B4 --> BG
BG -- 否 --> B6 --> B2
BG -- 是 --> G0 --> GG
GG -- toy --> G1 --> G3
GG -- llama2 --> G2 --> G3
G3 --> V1 --> VG
VG -- 是 --> B4
VG -- 否 --> G4 --> End((方法推理完成)):::terminal

T1 --> T2 --> T3 --> T4
T4 -. 训练得到参数 .-> G1
T4 -. M3权重/策略 .-> G2

subgraph Notes["旁注（详细解释）"]
direction LR
N3D["3D模型真正作用点：B2/B3。<br/>没有它，citation失去空间语义，grounding会塌。"]:::note
NLLM["LLM真正作用点：G2 + T3。<br/>G2负责推理生成；T3负责学合同/拒答。"]:::note
NSW["后端开关：GG。<br/>同一套上游token证据，可切toy或llama2。"]:::note
NAUD["审计闭环：V1→VG。<br/>不通过就回到BET细化，不直接放行。"]:::note
N3D --> NLLM --> NSW --> NAUD
end

G2 -.-> NLLM
GG -.-> NSW
V1 -.-> NAUD
B2 -.-> N3D
```

## 2) LLM 作用位地图（最容易混淆的点）

| 位置 | 是否必须 | 作用 | 相关代码 |
|---|---|---|---|
| `G2` 推理端 | 可选 | 把 token 证据转成 `frames + text`，并输出 citations/q/refusal | `provetok/pcg/llama2_pcg.py` |
| `GG` 后端开关 | 必须 | 决定走 `toy` 还是 `llama2` 生成分支 | `provetok/experiments/run_baselines.py` (`pcg_backend`) |
| `T3` 训练端 M3 | 可选 | 在训练阶段学习合同约束/拒答策略 | `provetok/training/stages.py` |
| `T4 -> G2` | 条件依赖 | 只有开启 M3 / LLM 训练时才把该参数策略注入 LLM 生成分支 | `provetok/training/trainer.py`（stage config 驱动） |

## 3) 一句话回答“LLM 到底是不是核心”

- 对于这个项目，**3D 证据链（BET+Verifier）是底座**，LLM 是可切换的生成后端。  
- 没有 LLM，系统仍可跑完整证据闭环；没有 3D 证据链，LLM 只能“说得像”，很难满足 grounding 审计。  
