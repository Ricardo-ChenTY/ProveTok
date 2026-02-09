你这篇稿子其实已经把 **Experiments** 设计成“claim-level proof suite”：主张用统一预算 (B=B_{enc}+B_{gen}) 的协议（BET+PCG+verifier+refusal），在 **gold-mask real profile** 上做主结论，在 **silver-label profile** 上只做 stress tests（明确写了“not a replacement for gold evidence”）；统计上用 paired bootstrap（典型 nboot=20,000）+ Holm 校正，并把 budgets / counterfactual variants 当 family 处理。Table 2/3 已经覆盖 C0001–C0006 + V0003 的“最小口试证据集”。

现在你要解决审稿人那句“更像框架/协议贡献、算法机制不够展开”，**不是再堆更多相似曲线**，而是补两类“不可替代证据”：

1. **BET 的可发表算法性质（近似/VOI/次模覆盖）+ 近似最优证据**（把 BET 从“支持多 family”变成“我们提出一个budgeted selection算法”）；
2. **外部有效性（协议/机制可迁移）**：换生成器/换 backbone/换数据源仍成立，并且 **阈e 还是 ratio。
* **E0.2 Budget 会计表（新增 Table Sx）**：
  你已经强调 matched-cost、固定 budgets（(2\times10^6) 到 (7\times10^6)，6个点）。把每个 budget 下：
  evidence tokens 数、平均分辨率 level、verifier 调用次数、refine 次数、p95 latency 组成，列成表（附录即可）。
  **证明**：budget coupling 真的是“可复核会计”，不是口号。

> 这两项写好，后面所有“新实验”才不会被认为建立在沙滩上。

---

### E1 BET 算法化 + 理论性质 + 近似最优证据（最高优先级，直接拆“协议贡献”）

你现在的 BET 只写“支持多 family（fixed-grid/ROI/scored）”，这是导致“像接口层”的根因。

#### E1.1 新增一个 **BET-Alg（ours）**：cost-aware greedy / coverage 目标

**要证明什么**：在相同 (B_{enc}) 下，**你提出的选择算法**比固定网格/ROI/简单 topk 更能提升 grounding 与 supportedness（同时满足 latency gate），且这种提升不是 claim：**C0007 (BET near-oracle gap)**

  * Pass 条件：在 6 个 budgets 中 ≥5/6 的 gap < 10–15%（阈值你自定，但要写死）

> 这一步就是你前面说的“给 BET 可发表性质”：你可以在方法里给出次模/VOI解释，在实验里给出“经验近似最优”证据——对 MICCAI 很吃香。

---

### E2 如果你确实在用 LLaMA2：必须做“约束消融 + 生成器可替换”

你的 PDF 当前 Method/Experiments 并没有明确写“PCG 用哪种生成器”，只写“生成 frame + citations”。如果你实际用的是 LLaMA2，那么你必须把它变成**“协议可迁移的 backend”**，否则接了 LLM 也不会加分。

#### E2.1 LLM 约束消融（同一个 LLaMA2，不同约束）

**要证明什么**：收益来自 **proof-carrying contract + verifier/refusal**，而不是“用 LLaMA2 本身”。

建议四档（同预算 sweep）：

1. **LLaMA2 free-form**：只提示生成报告，不强制 citations
2. **+ schema only**：强制输出 fraLLaMA2 trick”，而是“协议可迁移”。

* **设置**：固定 BET、verifier、refusal calibration；**(\tau_{refuse}) 只在 dev 上选一次并冻结**（你已承诺）
* **对照**：

  * PCG=LLaMA2（当前）
  * PCG=一个更小LM/不同LM/非LLM结构化生成器（任选一个即可）
* **输出**：复现 C0003 的 counterfactual non-triviality（no-cite / cite-swap / ω-perm family） 和 C0005 refusal gates
* **产出**：新增验证 claim：**V0004 (Generator portability with frozen calibration)**

> 回答你的“是不是还要跑 LLM”：**要跑，但只需要做 inference 级别的 ablation/swap**；不需要训练新 LLM。关键是把 LLM token/调用次数计入 (B_{gen})（放进 E0 的 budget 会计表）。

---

### E3 外部有效性：换 backbone / 换数据源仍成立（你现在只有一半）

你现在已经做了 cross-dataset silver stress tests，并且 V0003 用 pooled ω-perm 证明 citation channel 非装饰；但你自己也在 Limitations 里写了 sil*：新增验证 claim：**V0005 (Backbone transfer under matched-budget)**

#### E3.2 External gold subset（强烈推荐：把 future work 变成已完成）

你已经把“扩展 cross-domain evaluation with gold-mask subsets”写进 future work。如果你能做一个很小的 external-gold 子集（哪怕 20–50 cases、只标 2–3 个 critical findings 的粗mask），这会把外部有效性从“stress test”提升到“有 gold 支撑的迁移证据”。

* **产出**：新增 profile + 复现 C0004/C0005；作为 **V0006 (External-gold transfer)**

---

## 2) “还要不要跑 LLM？”——结论与最小跑法

* **如果 PCG 里确实用了 LLaMA2**：要跑，但只跑 **E2.1（同LLM约束消融）+ E2.2（generator swap）**。这是把 LLM 从“背景噪音”变成“外部有效性证据”的关键。
* **如果 PCG 其实没用 LLM**：也没问题，你就把 E2 换成 “PCG backend swap（两个非LLM生成器）”。核心是“协议可迁移”，不是“必须LLM”。tual（V0003）
* 如果你加 external-gold subset，就在这里新增一个 profile 行，并说明“只标 critical finding masks，用于外部有效性”。

**本节想证明**：你的主结论建立在 gold evidence 上，silver 只做鲁棒性；这与你的 Limitations 一致。

---

### 4.2 Baselines and Model Variants（你现在写“tokenization strategies”，不够）

把变体分两组写清：

1. **BET 家族**：fixed-grid / ROI / scored / **BET-Alg(ours)**（新增）
2. **PCG/合同家族**（如果你用 LLaMA2）：free-form / +schema / +verifier / full（E2.1）
3. **Backbone**（E3.1）：A/B 两种 encoder-decoder（例如 CT2Rep-Strong）

**本节想证明**：你不是“只有一个系统”，你有可定位的机制消融，能支撑“这是算法/机制贡献”。

---

### 4.3 Matched-Budget Protocol and Cost Accounting（你已有，但要补会计细节）

你已经写了：固定 budgets (2\times10^6..7\times10^6) 六点、FLOPs/latency matched、并且校准何尺度上算），以及 unsupported 的定义是否是 fraction（与文字一致）。

**本节想证明**：你的指标是“auditable”，与 claim-level proof rules 一致。

---

### 4.5 Statistical Protocol（保持你现在的写法，再补一句“family定义”）

你已经写了 paired bootstrap（nboot≈20k）、one-sided、Holm over family。
建议补一句：新加入的 E1/E2/E3 各自的 family 怎么定义（比如 “budgets×backbones” 作为一个 family）。

**本节想证明**：你不是 cherry-pick，而是把多预算/多变体当作 family 进行控制。

---

### 4.6 NEW: BET Optimality and Theoretical Property (E1)

写三段就够：

1. 定义 BET-Alg 的目标函数（coverage/VOI/submodular 任选一条）
2. 给 Algorithm 1（边际增益/成本贪心）
3. 给 near-oracle gap 的实验设置与 pass 条件（C0007）

**本节想证明**：BET 是算法贡献（“可发表性质”+“近似最优证据”）。

---

### 4.7 NEW: External Validity (E2+E3)4**：grounding IoU 多预算 Holm 显著（你已有）
* **C0005**：refusal calibration + anti-silencing gates（你已有）
* **V0003**：cross-dataset pooled ω-perm（你已有）
* **新增 C0007**：BET near-oracle gap（E1.2）
* **新增 C0008**：同一 LLM backend 的合同消融（E2.1）
* **新增 V0004**：generator swap + 冻结 calibration（E2.2）
* **新增 V0005/V0006**：backbone transfer / external-gold transfer（E3）

---

如果你按上面补齐：

* **E1（BET算法+gap）** 会直接把“框架贡献”拉成“算法+机制”；
* **E2/E3（可迁移）** 会把“协议”从“宣言”变成“可外推的科学结论”；
* 并且你现有的 claim-level proof gate/脚本化审计（proof_check/oral_audit）还能直接扩展到新 claims（你在 Reproducibility 已把这条路写明）。

如果你愿意，我可以把 **4.6/4.7 两节**按你论文口吻写成“可直接粘贴进正文”的英文段落（含你新增 claims 的 pass/fail 句式与统计 family）。
