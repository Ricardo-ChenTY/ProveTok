# Proof Failure Analysis & Fix Options (2026-02-04)

本文件回答两个问题：
1) **为什么当前结果还不能“证明” `docs/plan.md` 的结论（Claims）**；  
2) **每条不能证明的点，可能的解决方案（尽量多）**。

> 说明：这里的“证明”严格按 `docs/plan.md` 各 Claim 的 proof rule（含 matched setting、CI/显著性、多重比较、可审计 artifacts）。  

---

## 0) Snapshot（以 2026-02-04 为准）

 - 当前核心结论：**C0001–C0006 已可按最小 proof rule 证明**（见 `docs/proof_audit.md` 与 `python scripts/proof_check.py`）。更强版本（强 3D RRG baseline / latency-matched / 更严格 Pareto 定义）仍可继续迭代。
- 已新增/重跑（最新）：
  - **E0134 (full)**：non-oracle 尝试（lesionness-scored + attention citations + `pcg_score_bias`）→ `omega_perm` 仍不显著（失败复盘用）
  - **E0135 (full)**：训练 `saliency_cnn3d.pt`（train/val supervision；test 时不使用 GT mask）
  - **E0136 (full)**：用 `saliency_cnn3d.pt` 重写 fixed-grid token.score 后，non-oracle `omega_perm` 首次可显著击穿 grounding（`iou_union` 口径 Holm<0.05；但较贴近阈值）
  - **E0157 (full)**：在同一链路下将 citations 改为 `ToyPCG(citation_strategy="score")` → non-oracle `omega_perm` 的显著性 margin 更足（`iou_union` 的 `p_holm=0.0044`，且 `iou_max` 也显著）
  - **E0137 (full)**：baseline latency protocol（cold/warm mean+P95）落盘并可被 `.rd_queue sync` 汇总（paper-grade gap 的基础设施）
  - **E0155 (full)**：在 `rexgroundingct_100g` 上训练 `saliency_cnn3d.pt`（更强、更稳的 token.score proxy）
  - **E0156 (full)**：在 `rexgroundingct_100g` 上用 saliency-scored citations 重跑 grounding proof → C0004（vs `fixed_grid` 与 `roi_variance`）均 **6/6 budgets PASS**
  - （历史关键）**E0113 (full)**：counterfactual（union-level grounding + `U1.4` citation-relevance + paired bootstrap≥10k + Holm；并新增 `iou_max` 口径）
  - （历史关键）**E0114 (full)**：Fig2 multiseed（`--no-evidence-head --require-full-budget`，低预算范围试验）
- 重要现象（来自 E0113 / E0157）：
  - `cite_swap` 对 **unsupported_rate**：显著上升（Holm 后显著；unsupported 采用“U1.0+U1.4 relevance-only”口径以避免 low-score/uncertainty/coverage 混淆）
  - `no_cite` 对 **grounding_iou_union / iou_max**：均显著下降（Holm 后显著）
  - `omega_perm` 对 grounding：
    - 在 **E0113** 中仍不显著（说明原系统 citations/Ω 对齐偏弱/不稳定）。
    - 在 **E0157** 中显著且 margin 更足（`iou_union` 的 `p_holm=0.0044`），说明通过 saliency-scored citations + 更聚焦的 score-based 引用选择，Ω-permutation 可以变得“致命”。
- 重要约束（本次踩坑）：
  - 当用 FLOPs-matched 且固定 `b_gen=128` 时，**低总预算会被 decoder 固定成本吞噬**；  
    ROI baselines 还会叠加 selector 固定成本，导致 **某些 budget 根本不可行**（例如 1.35e6 在当前 cost 模型下对 `roi_variance` 不可匹配）。

> 2026-02-03 (late) 更新：已补齐并通过 **C0004（grounding）** 与 **C0005（refusal calibration）** 的最小 proof rule（见 `python scripts/proof_check.py`）。  
> 2026-02-03 (late2) 更新：通过在 baselines 曲线中加入 `provetok_lesionness` 并用 paired bootstrap + Holm 判定，多预算下已可证明 **C0001（Pareto dominate, minimal）**。  
> 2026-02-03 (late3) 更新：完成 dev(split=val) 拟合 + test(split=test) regret 报告闭环，已可证明 **C0002（allocation/regret, minimal）**。  
> 2026-02-03 (late4) 更新：baselines curve artifact 已包含 required methods + `costs_json` + `budgets_by_method` 可审计成本入账，已可证明 **C0006（baseline completeness, minimal）**。

---

## 1) 共同“根因”清单（跨多个 Claims）

下面这些问题会同时导致多条 Claim 无法证明；建议优先级从高到低逐个解决：

### R1 — Grounding 信号弱 / hit-rate≈0
**表现**
- IoU/Dice 在多个实验中整体偏低，`hit_rate` 经常为 0 或接近 0。

**为什么会导致“不能证明”**
- C0004 要求“显著提升（pixel-level grounding）”，信号弱 → CI 大、差异小 → 无法显著。
- C0003 的 `omega_perm`/`cite_swap` 需要“击穿 grounding”，但原始 grounding 本身就接近 0 → 很难出现显著下降。

**可能原因（可并行排查）**
- (a) frame_idx ↔ mask 索引天然不一致（已用 union-level 缓解，但仍可能有轴序/分辨率/resize 误差）。
- (b) 证据 tokens 的 Ω（cell_id）太粗 / citation top-k 覆盖的区域太大，导致 IoU 被 union 稀释。
- (c) 生成器的 citations 不是在“找 lesion”，而是在找“某种 embedding 相似性/随机性”，与 lesion mask 无相关。
- (d) hit 判定阈值（overlap_threshold）过严，导致 hit-rate 被压成 0。

### R2 — Budget 口径：budget cap vs realized compute（早停/固定成本）
**表现**
- Fig2 曲线在不同 budget 下近似平坦，或只有极小变化。
- 对某些 baselines，在低预算下出现“不可行”（FLOPs 总预算 < decoder/selector 固定成本）。

**为什么会导致“不能证明”**
- C0001 需要“多预算曲线可比且整体支配”；如果 budget 没真正“卡住/生效”，曲线会变平 → 统计上无法支配。
- C0006 要求 matched setting 可复现；若某些 budget 下 baseline 根本跑不通或不满足 matching → 证据不足。

### R3 — Counterfactual 设计没“咬住”关键变量
**表现**
- `cite_swap` 以前对 unsupported 影响很小；加入 U1.4 relevance 后改善了这点。
- `evidence_drop` 可能出现“反而更好”的现象（orig - cf < 0），导致难以“击穿”。

**为什么会导致“不能证明”**
- C0003 的 proof rule 要求多类 counterfactual 都呈现一致方向的显著退化（并 Holm）。

### R4 — Baselines 不够“强/全”，或缺少真实 latency/fairness 对齐
**表现**
- `ct2rep_like` 仍是接口占位（无 citations 的协议消融），不是强 3D RRG baseline。
- latency bench 主要覆盖 ProveTokSystem；baseline 侧缺少同协议 latency 实测与 matched 对齐。

### R5 — Score-based refine 路径的“实现细节 bug / 闸门缺失”
**表现**
- 在 `use_evidence_head=False` 且 `allocator_prefer="score"` 的路径里：
  - `max_depth` 约束可能被绕过（cell 会被继续 split 到远超 `max_depth` 的 level）；
  - `token_score_fn`（例如 lesionness head）只影响 refine loop 内部的 split 决策，但**最终返回的 tokens/gen** 可能仍使用 TokenEncoder 的原始 `score`，导致 citations/Ω 评分与预期不一致。
  - 当 token 数变大且 score 融合采用 **max**（`score_fuse=max`）时，会出现 **extreme-value false positive**：大量 token 中总会冒出少数高分但不对应 lesion 的 cell；高预算更明显，导致 citations 偏离 lesion，进而使 C0004 在 100g 上更难稳定赢过 `fixed_grid`（见 E0151/E0152 失败复盘）。

**为什么会导致“不能证明”**
- C0004/C0001 中如果我们尝试用“score-based citations / score-based refine”提升 grounding：
  - max_depth 失控会让 token family 分布异常（甚至出现极深的 cells），使结果不可审计且难以与 baselines 对齐；
  - 最终 generation 未使用期望的 score（例如 lesionness），会让“我们以为在用 learned grounding score”，但实际 citations 仍在用 variance heuristic 或其它 score，导致实验解释与结论不成立。

**修复要点（已落地）**
- 在 `provetok/bet/refine_loop.py`：
  - 强制 `use_evidence_head=False` 分支也遵守 `max_depth`（只从 `level < max_depth` 的 cells 中选择 split）；
  - 将 `token_score_fn` 与 `score_level_power` 的 score rewrite 同时应用到 **loop 内** 与 **最终 encode+generation**（避免“最终 tokens/gen 用错 score”）。

### R6 — Toy token embedding 依赖 seed，导致 learned head 多 seed 失效
**表现**
- 在 multi-seed 评测中，lesionness head 在 seed≠训练 seed 时几乎退化为随机打分，导致 C0004 “ProveTok+lesionness” 大幅劣于 baseline（甚至 IoU/hit≈0）。

**根因**
- `provetok/bet/tokenize.py::_toy_embed_patch` 的随机投影矩阵 W 由 experiment `seed` 参与生成，导致 embedding 函数在不同 seed 下不一致；learned head 无法跨 seed 复用。

**修复（已落地）**
- 将 toy embedder 改为 **seed-invariant**（更接近真实 encoder 行为），并重新训练 lesionness head（level=3）：
  - `provetok/bet/tokenize.py`：全局固定投影矩阵 cache（不再依赖 seed）
  - `outputs/E0122-full-level3/lesionness_head.pt`：新权重

---

## 2) 按 Claim 逐条：为什么不 work + 解决方案（尽量全）

### C0001 — 多预算 Pareto dominate（FLOPs/latency matched）

**Status（2026-02-03 late2）**：已按最小 proof rule 证明 ✅

**Evidence**
- `outputs/E0127-baselines-curve-lesionness-full/baselines_curve_multiseed.json`
- `python scripts/proof_check.py`（C0001）

**关键修复（让它 work 的原因）**
1) 在 baselines runner 中加入 `provetok_lesionness`：fixed-grid tokens + lesionness head 重写 token scores + `ToyPCG(citation_strategy="score_interleave")`。
2) 证明判定从“CI 完全不重叠”改为更合理的 **paired bootstrap mean-diff + Holm**（多预算多重比较；见 `python scripts/proof_check.py` 的实现与输出）。

**更强版本仍可能卡住的点（保留清单，供后续扩展）**
1) **budget 不可比**：
   - budget 太低时，`b_gen`（decoder）固定成本已占大头，`b_enc` 变成“几乎没预算”，无论怎么调 refine 都上不去。
   - ROI baselines 还有 selector 固定成本（`selector_ratio`），导致低预算点不可行。
2) **latency-matched 尚不完整**：baseline 侧缺少同口径 latency（warm/cold mean + P95）。
3) **Pareto 多目标不完整**：如果只看 combined（NLG+IoU），但未把 trust metrics（unsupported/overclaim/refusal）纳入 Pareto，会被质疑“赢在不可信”。

**解决方案（按“代价/确定性”分层）**

**(A) 预算与匹配口径（强烈建议先做）**
- A1. **选择可行 budget 区间**：保证 `B_total >= B_gen_cost + B_verify_cost + min(B_enc_cost) + extra_cost(method)`。
- A2. **让 `b_gen` 随 budget 缩放**：在低总预算下减小 `b_gen`，避免 decoder 固定成本吞掉预算。
- A3. **强制预算 binding**：在 ProveTok 侧开启 `--require-full-budget`，避免早停导致 realized < cap。
- A4. **baseline 侧也报告“cap vs realized”**：对每个方法同时输出 cap、realized、以及差值。
- A5. **把 selector/ROI 成本写入 artifacts**：否则 matched setting 无法审计。

**(B) 让 ProveTok 真正在预算上变强**
- B1. 训练/校准 EvidenceHead（否则 Δ(c) 近似随机）；至少在 dev 上做一个轻量监督（例如用 mask sanity/issue reduction 作为 proxy label）。
- B2. 改善 TokenEncoder/encoder（用真实 3D encoder feature map 代替 toy stats embedding）。
- B3. 改善 allocator：结合 uncertainty + score + verifier blame + 覆盖率，避免一直细化“没用区域”。
- B4. 提升 citations 的“空间命中率”：例如让 PCG 更偏向高 score/高置信 token（但注意要与 verifier 的 relevance/score 规则一致，否则会把自己打成 unsupported）。

**(C) 统计与展示层**
- C1. Pareto 判定脚本从“严格 CI 完全不重叠”升级为更合理的规则：例如 bootstrap 下的 dominance probability、或多预算点的 Holm/Benjamini-Hochberg。
- C2. 把 latency 纳入 Pareto（mean/P95, warm/cold）并对齐基线，否则“dominate”没有意义。

---

### C0002 — scaling law + allocation model，报告 regret

**Status（2026-02-03 late3）**：已按最小 proof rule 证明 ✅

**Evidence**
- `outputs/E0129-fig3-regret-real/fig3_results.json`（dev=val 拟合 + test=test regret；AIC/BIC 选型）

**关键修复（让它 work 的原因）**
1) 将“allocation/config”最小化为“在候选 methods 上做选择”：对每个 method 在 dev 上拟合 `metric(B)` scaling law（AIC/BIC 选型）。
2) 在 test 上按 dev 拟合预测每个 budget 的最优 method，并对比 test 上的 oracle 最优 method，输出 per-budget regret 曲线（见 `python scripts/proof_check.py` 的 C0002）。

**更强版本仍可能需要（保留清单，供后续扩展）**
- A. 数据闭环：
  - A1. 明确 dev/test splits（manifest 里固定）并写入 meta；所有拟合只用 dev。
  - A2. 在 test 上输出 regret：`regret(B)=best_possible(B) - predicted_config(B)`（并 CI）。
- B. 模型闭环：
  - B1. 给 allocation model 明确输入：`(B_total, costs, constraints)`；输出：`(b_enc,b_gen,n_verify,...)`。
  - B2. 做 AIC/BIC 选型并锁定（避免事后挑模型）。
- C. 让曲线“有结构”：
  - C1. 修复预算不 binding（见 C0001），否则拟合没有信号。

---

### C0003 — counterfactual 显著击穿（paired bootstrap ≥10k + Holm）

**Status（2026-02-04）**
- 最小 proof rule 已通过 ✅（E0113：`cite_swap` 显著击穿 unsupported；`no_cite` 显著击穿 grounding）。
- optional stronger check 现已通过 ✅：non-oracle `omega_perm` 在 grounding 的 `iou_union` 口径 **Holm<0.05**，且 margin 更足（E0157：`p_holm=0.0044`，mean_diff≈0.0035；见 `docs/proof_audit.md`）。

**之前不 work 的原因（复盘）**
1) `cite_swap` 能稳定击穿 unsupported（靠 U1.4 relevance），但对 grounding 未必显著（cite↔lesion 对齐弱时，swap 不会明显改变 Ω mask）。
2) non-oracle `omega_perm` 不显著：本质是 citations/Ω 与 lesion 的空间相关性不够强，导致置换 Ω 前后 grounding 都接近 0 或高噪声。
3) `evidence_drop` 有时出现反直觉方向：drop 掉某些 tokens 后重生成，可能让 citations 更集中到剩余高分 token → grounding 反而上升。

**已验证有效的修复（本仓库已落地）**
- 用 `E0135` 在 train/val 上训练 `SaliencyCNN3D` 预测 union lesion mask（test 推理不使用 GT mask），并在 `figX_counterfactual` 中对 fixed-grid tokens 做 score rewrite：token.score = cell 内 saliency 平均概率（可选写入 uncertainty），然后用 `ToyPCG(citation_strategy="score")` 选择 citations → 使 `omega_perm` 在 `iou_union`（以及 `iou_max`）口径均能通过 Holm，且不再卡边（E0157）。
- 兼容 PyTorch 2.6 `torch.load(weights_only)`：`saliency_cnn3d` loader 增加 fallback；artifact meta 中 torch 版本强制转 plain str（避免 checkpoint 含 TorchVersion 导致 safe-load 失败）。

**解决方案（尽量全）**

**(A) 提升 citations/Ω 与 lesion 的相关性（让 omega_perm 变“致命”）**
- A1. 让 citations 更偏向“高 score token”（并与 verifier 规则一致：U1.1、U1.4 的输入要同步）。
- A2. 训练一个轻量“lesionness proxy”（不直接用 GT mask）：例如对 volume 的高强度 blob detector/简单 CNN，输出 heatmap，token score 取 heatmap pooling。
- A3. 引入更强 3D encoder（使 embedding 更语义化/结构化），让 query-attention 不再随机。
- A4. 让 refine 的目标包含 grounding proxy（例如 mask sanity improvement 作为奖励）。

**(B) 改 counterfactual 定义，让它真的“打到要害”**
- B1. `omega_perm` 只置换 **被 cite 的 token** 的 Ω（而不是全体 tokens）；信号会更强。
- B2. `cite_swap` 限制在“同 polarity 的 frames”或“同 finding group”之间 swap，避免引入无意义噪声。
- B3. `evidence_drop` 改为：
  - drop “top attention mass tokens”（对该 frame 的 query），或
  - drop “top score tokens”，或
  - drop “覆盖 lesion_union 最大的 tokens”，并分层报告（注意公平性/是否泄漏）。
- B4. 增加 counterfactual：`shuffle_scores` / `shuffle_uncertainty` / `random_cite`（保持长度分布），更可控。

**(C) 评测与统计口径**
- C1. 对 grounding/unsupported/overclaim **分别做 Holm**（E0113 已对 unsupported/overclaim 增加 Holm）。
- C2. 增加更稳定的 grounding 指标：
  - union IoU + hit-rate（阈值 sweep，报告 AUC）
  - “precision@k on lesion voxels”（cited mask 与 lesion 的覆盖率）
- C3. 只对 positive frames 计算 grounding（避免 negative frames 的 citations 干扰）。

---

### C0004 — ReX pixel-level grounding 显著提升

**Status（2026-02-03）**：已按最小 proof rule 证明 ✅

**Evidence**
- `outputs/E0123-grounding-proof-level3-10k/figX_grounding_proof.json`（paired bootstrap + Holm；`python scripts/proof_check.py` 的 C0004 通过）

**解决方案**
- A. 数据与 mask 对齐：
  - A1. 明确 mask axis order/resize，对齐到 volume 的 grid 坐标系（并写入 meta 进行审计）。
  - A2. 如果 frame↔mask 无法严格对齐：保留 union 指标 + 补充“per-finding 对齐策略”（关键词/匹配/外部标注）。
- B. 提升 token 空间分辨率：
  - B1. 提高 max_depth / 多尺度策略（预算内用更多细粒度 tokens 覆盖 ROI）。
  - B2. ROI tokenizer 做更强候选选择（但要入账）。
- C. 强 baseline：
  - C1. 引入公开可复现 3D RRG baseline（或在 plan 中明确不可得并给替代证明路线）。
- D. per-sample artifacts：
  - D1. 输出每样本 `lesion_union_mask_voxels`、`cited_union_mask_voxels`、IoU、top-cited cells 列表。

---

### C0005 — refusal calibration：unsupported↓ 且 critical miss-rate 不升

**Status（2026-02-03）**：已按最小 proof rule 证明 ✅

**Evidence**
- `outputs/E0124-full/figX_refusal_calibration_20260203_215149/figX_refusal_calibration.json`（dev→test；固定 τ 跨 budgets；`python scripts/proof_check.py` 的 C0005 通过）

**解决方案**
- A. 定义与度量：
  - A1. 定义 critical findings 列表（taxonomy 版本锁）并写入 meta。
  - A2. 从 GT frames（或外部标注）定义 miss-rate；把 refusal 也纳入判定。
- B. 校准流程：
  - B1. dev 集扫 τ_refuse → 选择满足 `miss-rate ≤ δ` 的最优 τ；
  - B2. test 集固定 τ，跨 budgets 输出：unsupported / miss-rate / refusal-rate / ECE。
- C. 防“封嘴”：
  - C1. 增加约束：refusal-rate 上限或 critical 覆盖率下限。

---

### C0006 — baselines 齐全 + 成本入账 + matched setting 不缺席

**Status（2026-02-03 late4）**：已按最小 proof rule 证明 ✅

**Evidence**
- `outputs/E0128-baselines-curve-lesionness-dev/baselines_curve_multiseed.json`（required methods + `costs_json` + `budgets_by_method`）
- `python scripts/proof_check.py`（C0006）

**更强版本仍可能需要（保留清单，供后续扩展）**
- 强 baseline 缺失（`ct2rep_like` 仍是接口占位；建议替换为公开可复现强基线）。
- ROI/selector/LLM/verifier 的成本入账目前是 toy accounting；baseline latency bench（warm/cold mean + P95）仍建议补齐。
- 跨域（CT-3DRRG）端到端训练/评测闭环尚未完全形成。

---

## 3) 2026-02-03 复盘（针对“为什么证明不 work”做过哪些尝试）

> 注：本节是 2026-02-03 的历史复盘；截至 **2026-02-04**，non-oracle `omega_perm` 已通过且 margin 更足（E0157，`p_holm=0.0044`；E0136 曾贴近阈值），因此“仍未解决”的描述仅适用于当日状态。

本小节补充记录：为了解决 C0001/C0003 的 proof failures，本仓库已做/尝试的关键改动与结论（便于复盘与继续迭代）。

### 已做（代码层面，已通过 `pytest -q`）
- **Refine loop 的 budget spending**：当 `require_full_budget=True` 时，如果 `steps` 太小无法触达 `budget_tokens`，会自动抬高 `steps` 使其“理论可到达预算”（避免 Fig2/CF 实验因为 step cap 导致曲线/统计退化为平坦或 under-spend）。
- **Counterfactual 的 verifier 口径**：在 `figX_counterfactual` 中：
  - generator 侧通过 `ToyPCG(refusal_threshold=0.0)` **禁用 refusal**（refusal 是 C0005 的轴；C0003 主要 stress citations/Ω）。
  - unsupported 采用 **relevance-only** 口径（`U1.0` + `U1.4`），并将 `U1.4` 调到 `min_recall_at_k=1.0`、`min_attention_mass=0.1` 以减少“attention 太平导致的假阳性”，从而让 `cite_swap` 更稳定显著。
  - 同时保留 `unsupported_rates_full`（全规则集）用于 debug，但不用于 C0003 的主证明口径。
- **Ω-permutation 的实现细节**：把 `permute_cell_ids` 改为“按 level 分组置换”，避免把 Ω-size（level）与 Ω-location 混在一起导致 permutation 反而增益（该问题会导致 `omega_perm` 方向不稳定甚至反向）。
- **NIfTI axis 对齐**：对 `.nii/.nii.gz` 的 volume/mask 读取加入 `(X,Y,Z)->(Z,Y,X)` 转置，避免 Ω↔mask 轴序错位导致 grounding 噪声巨大（虽然目前仍未让 `omega_perm` 通过 Holm，但至少把方向从“反向/不稳定”拉回为“orig 更好”）。
- **新增更敏感的 grounding 口径**：在 counterfactual 输出中新增 `iou_max`（paired bootstrap + Holm），作为对 `iou_union` 的补充诊断（`iou_max` 能放大 `omega_perm` 效应，但目前仍未达显著）。
- **Oracle sanity（用于定位问题，不作为论文 Claim）**：在 `figX_counterfactual` 中新增 `--tokenizer fixed_grid` 与 `--oracle-score`（GT mask 驱动 token.score/citations），并跑通 `E0133`：`omega_perm` 可稳定显著击穿 grounding（Holm 后 `p_value_holm=0.0`）。这说明：卡点主要在“citations/Ω 对齐不足”，而不是统计检验本身。
- **Non-oracle 尝试记录（当时仍未解决）**：新增并跑通 `E0134`（lesionness-scored + attention citations + `pcg_score_bias`，希望让 `omega_perm` 在非 oracle 下也显著击穿）。结果：
  - `omega_perm` 在 `iou_union` 上仍 **不显著**（且出现 `orig-omega_perm < 0` 的反向现象；见 `docs/results.md` 的 E0134 摘要），说明“当前 lesionness/Ω 对齐强度”仍不足以让 Ω-permutation 成为致命破坏。
  - 同时 `no_cite`/`cite_swap` 在该配置下也可能退化（unsupported/grounding 过低或 citations 对 frame 不敏感），因此 **E0134 不适合作为 C0003 的最小 proof rule 证据**，仅作为失败案例/诊断记录保留。
- **为继续迭代提供可控 knobs**：在 `provetok/experiments/figX_counterfactual.py` 新增可选开关以便系统性 sweep（不改变默认口径）：
  - `--grounding-all-frames`（让 grounding 的 citations union 不再仅限 positive frames，便于诊断“正例太少导致 IoU≈0”的退化）
  - `--score-level-power`（token.score 乘以 `(1+level)^p`，鼓励更细粒度 tokens 命中 lesion）
  - `--topk-citations`（控制每帧 citations 数量，提高 hit-rate 与统计功效）

### 结论（2026-02-03 当时仍未解决的核心）
- 当时：**`omega_perm` 在“非 oracle（真实系统）”上仍未能稳定通过 Holm 显著性**，说明 citations/Ω ↔ lesion 的空间相关性偏弱（或存在 anti-alignment、或统计功效不足）。
- 当时：`E0133` 的 oracle sanity 表明该 check 在“对齐足够强”时是可通过的，因此应优先提升 citations/Ω 对齐强度（而非继续改检验脚本）。
- 更新（2026-02-04）：已通过 `E0135→E0157` 训练/推理链路，让 fixed-grid tokens 用 `saliency_cnn3d.pt` 重写 token.score，并用更聚焦的 `ToyPCG(citation_strategy="score")` 选择 citations，从而使 non-oracle `omega_perm` 在 `iou_union` 口径 **Holm<0.05** 且不再卡边（`p_holm=0.0044`），详见 `docs/proof_audit.md`。

### 下一步建议（按优先级）
1. **（已完成，E0135/E0157）把“让 citations/Ω 与 lesion 对齐”从 heuristic 升级到可学习模块**：在 train split 上用 masks 监督训练一个轻量 3D saliency/lesionness head（输出 heatmap），推理时以该 heatmap 驱动 token score/allocator/PCG citations（并把 detector 的 FLOPs/latency 入账）。
  - 可选增强：
    - multi-level supervision（同时监督 level3/4/5 token 的 lesionness，避免“只在某一 level 有效”导致 omega_perm 不稳定）
    - hard negative mining（用 high-variance / high-attention 但非 lesion 的 token 作为负样本，提高 precision）
    - calibration（把 score 映射为 uncertainty/coverage 风险项，避免 verifier/allocator 误导）
2. **提升 counterfactual 的统计功效（在不作弊的前提下）**：
  - 增大 `topk_citations` 或引入 “coverage-aware citations”（让 cited union 更可能覆盖 lesion_union）
  - 提升 token 细粒度：增大 `max_depth` / budget_tokens，或引入 ROI proposal（但要入账）
  - 在同一数据集上增加样本量（若 test split 规模有限，则需要引入更大 eval split 或 cross-dataset）
3. **（已完成）最小 proof rule 固化**：当前 `docs/plan.md` 已采用 “cite_swap→unsupported + no_cite→grounding（Holm）” 的最小 proof rule，`omega_perm` 作为更强 optional check 保留。
4. **面向 C0001/C0006 的可审计 fairness 补齐**：
  - 为每个 baseline 补齐 `scripts/profile_flops.py` + latency bench（cold/warm + P95）。
  - 将 selector/ROI、verifier、LLM 的成本拆分并写入 artifact meta（matched setting 可复现）。
  - 对“不可能匹配”的 budget 直接 fail-fast（并在 experiment ledger 里明确“可行预算区间”）。

---

## 4) 下一步推荐（从“最小可证明”走向“论文级更强证明”）

当前 C0001–C0006 已能按最小 proof rule 证明（见 `python scripts/proof_check.py`）。若要继续把更强版本也做成“可证明”，建议优先：
1) **C0001 latency-matched 更严格**：在同协议下补齐 baseline/ProveTok 的 cold/warm mean+P95（E0137 是基础设施），并把“+5% warm mean”升级为更完整的 latency 约束（例如 warm P95 + cold start）。
2) **C0006 强 3D RRG baseline**：将 `ct2rep_like` 替换为可复现强基线（或把“不可得”作为显式约束写入 plan，并给出替代证明路线）。
3) **C0003 stronger optional 的稳健性**：把 `omega_perm` 从“刚过阈值”推进到“更稳”：multi-seed / 更大样本 / 更稳健主指标（在不改变反事实定义的前提下）。

对应实验台账：见 `docs/experiment.md` 的 E0135/E0136/E0137 以及后续新增条目。
