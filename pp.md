下面给你的是 **“可直接投 MICCAI 主文（v1.1）”** 的完整稿件（英文、8–9 页主文的写法风格），并且我把所有目前仍然“模糊/需要你落地拍板”的点 **全部单列出来**，然后给出我建议的默认选择（保证你可以不问我任何问题就开工实现/跑实验）。
同时我也按你的要求联网补齐了 **3D CT 报告生成/对齐** 的主要 baseline 线索与可引用事实，并在文中给出相应引用依据。([MICCAI Papers][1])

---

## A. 仍然模糊但必须在“开工前”定死的点（我已给默认值）

> 这些是 reviewer 最爱抓的“定义不清/复现不了/设定偷换”。我先把坑挖出来，再把默认值填好。你后续若改，也只需改这里和实验配置。

### A1. 3D tokenization 的几何定义（octree vs grid、level、预算）

**模糊点**：你 v0.2 里“CELLS / level / +7”像是 octree；但主文必须写清：cell 如何编号、如何保证跨 step 稳定、如何把 laterality 变成几何判定。
**默认选择（强烈建议）**：

* 用 **octree（8-way split）**：level ℓ 的 cell 数量为 (8^\ell)。
* **init_level = 1（8 tokens）**，**max_level = 3（512 tokens）**。
* split 一个 cell：删除该 cell，加入其 8 个子 cell → **token 总数净增 +7**。
* 预算 (B\in{64,128,256,512}) 表示 **任意 step 的 token 总数上限**。
* **coarse-only 判定**：正向断言若引用 token 全部来自 level ≤ 1，则视为 coarse-only（不够细）。

### A2. “证据是否够”的判定：要不要用 score/uncertainty

**模糊点**：你 proposal 里有 LowScore / uncertainty，但若 score 训练不可靠，reviewer 会直接说“自嗨指标”。
**默认选择（主文最稳）**：

* 主文的 **Hard verifier 只依赖可审计的几何与引用结构**（NoCitation / CoarseOnly / LateralityMismatch / BilateralSeparation）。
* **LowScore 不作为 hard-fail**，最多放成 *appendix* 的 soft signal（或仅用于 split heuristic 的 tie-break）。
* 这样可以避免 reviewer 把你整篇论文打成“靠一个不可靠分数在自证”。

> 你仍然可以在工程里保留 score，用来更好 split，但主文“承诺”要保守。

### A3. 报告单位：一句话 vs 一条 finding（verifier 的最小审计粒度）

**模糊点**：你说“每句/每个 finding 都要引用”。主文必须固定解析规则，否则无法复现。
**默认选择**：

* 以 **Finding 行** 为最小审计单元：每行必须有 `[CITE: ...]`。
* Findings 采用 “每行一个解剖/异常陈述”的模板，Impression 可复述但 **不允许引入新证据 id**（只可引用 Findings 出现过的 ids），防止“Impression 幻觉”。

### A4. “laterality” 几何判定基准（mid-sagittal plane 怎么定）

**模糊点**：CT orientation/affine 不统一会毁掉 laterality verifier。
**默认选择**：

* 预处理把体数据统一到 **RAS**（或你习惯的标准），并保证 x 轴对应 Left↔Right。
* midline 取体素空间 x 维中点（volume center plane）。
* token 的几何中心 (c_x)：若 (c_x < x_{\text{mid}}) 视为 **Left**，否则 **Right**。
* 对 “bilateral” 陈述：必须同时引用至少 1 个 Left token 与 1 个 Right token（BilateralSeparation）。

### A5. 数据集主战场：CT-RATE vs RadGenome vs CTRG-548K

**模糊点**：你 proposal 同时提 CT-RATE / RadGenome / CTRG-548K；主文必须“主线一致”。
**默认选择（最 reviewer-proof）**：

* **主实验：RadGenome-Chest CT（因为能评估 evidence grounding）**，它建立在 CT-RATE 上，并提供 197 类器官 mask、665K grounded report sentences、1.2M grounded VQA。([Nature][2])
* **补充实验：CT-RATE 报告生成（对齐 3D RRG 文献）**，并遵循 RRG-DPO 所述 split：train=24,128 volumes/20,000 patients；official test 再分 val=360/300、test=1204/1004（你可按同法复现）。([MICCAI Papers][3])
* **CTRG-548K（1,804 pairs）** 只做附录“可比性”实验：Dia-LLaMA 明确写了他们用 1,804 CT-report pairs（80/20 split）。([MICCAI Papers][4])

### A6. 你承诺的学习部分：IL 还是 DPO（二选一）

你要求“主文只承诺 heuristic vs learned split（IL 或 DPO 二选一），RL 放附录”。
**默认选择（更像 Best Paper 叙事）**：主文采用 **DPO 学 split policy**（不是 RL、但比 IL 强；而且你已有 verifier 作为偏好信号）。同时写清：这只是 **contextual bandit / one-step lookahead** 的 preference learning，不涉及长程 credit assignment → reviewer 不容易打成“RL 过度承诺”。
并且你可以在 ablation 里加一个“IL-only”作为补充（可选）。

---

# MICCAI Main Paper Draft v1.1

## Title

**ProveTok-Agent: Proof-Carrying and Budgeted Evidence Acquisition for 3D Chest CT Report Generation**

## Authors

*Anonymous for review.*

---

## Abstract

Radiology report generation for 3D chest CT has recently accelerated with multimodal foundation models and LLM-driven decoders, yet clinical deployment remains constrained by non-auditable hallucinations, laterality mistakes, and over-specific claims without spatial support. Prior 3D CT report generators (e.g., CT2Rep, Dia-LLaMA, recent 3D VLMs) primarily optimize textual or label-based objectives and seldom output **verifiable 3D evidence objects** that can be deterministically audited.([MICCAI Papers][1])

We propose **ProveTok-Agent**, a new inference-time paradigm for **proof-carrying** 3D CT reporting under a strict **evidence budget**. The system enforces a *proof contract* in which each finding is emitted with explicit citations to a deterministic hierarchy of 3D spatial tokens (ProveTok). A hard, deterministic verifier audits every finding for missing citations, insufficient spatial granularity, and laterality inconsistencies. When issues are detected, ProveTok-Agent performs **budgeted evidence acquisition** by selectively splitting spatial tokens (octree refinement) and rewriting the affected finding—mimicking the “zoom-and-confirm” workflow of radiologists—while respecting a fixed token budget. To improve acquisition decisions beyond heuristics, we further train a lightweight split-policy via **direct preference optimization (DPO)** on verifier-derived preferences.

We evaluate on RadGenome-Chest CT (grounded sentences with segmentation masks) and CT-RATE (paired CT volumes and free-text reports). RadGenome provides 197 organ masks, 665K grounded report sentences, and 1.2M grounded VQA pairs, enabling direct measurement of evidence grounding quality.([Nature][2]) We report (i) standard RRG metrics (BLEU/METEOR/ROUGE-L, CheXbert F1, RadGraph, RadCliQ, GREEN, RaTEScore) and (ii) new proof-centric reliability curves that quantify issue rate vs evidence budget.([ACL Anthology][5])

---

## 1. Introduction

Chest CT reporting is a high-stakes, high-volume clinical workflow. Unlike 2D chest X-rays, 3D CT demands slice-by-slice spatial reasoning and precise laterality, creating fertile ground for report generation failures: hallucinated positives, incorrect left/right, and overconfident statements unsupported by imaging. Meanwhile, recent 3D CT report generation methods—ranging from dedicated encoder-decoder architectures (CT2Rep) to LLM-driven CTRG systems (Dia-LLaMA, MvKeTR) and 3D VLM variants—primarily optimize textual similarity or label extraction scores, and do not produce *auditable spatial evidence objects* aligned to each generated statement.([MICCAI Papers][1])

A parallel trend in radiology generation emphasizes clinical alignment and factuality. For example, DPO-based optimization (RRG-DPO) improves clinical metrics on 2D and 3D datasets, but still evaluates the generated text as a whole rather than enforcing per-claim 3D evidence traceability.([MICCAI Papers][3]) Similarly, LLM-based evaluation metrics (GREEN) and structure-based metrics (RadGraph, RadCliQ, RaTEScore) improve assessment fidelity but do not yield an executable “proof object” attached to each finding.([ACL Anthology][6])

### Key idea

We argue that 3D CT report generation should be treated as **proof-carrying generation under a budget**: each emitted finding must carry explicit references to spatial evidence, and when evidence is too coarse, the system must actively acquire finer evidence (i.e., “zoom in”) while obeying a strict computational budget.

### Contributions (v1.1, deliberately narrowed)

We intentionally **collapse the main paper** to two hard contributions:

1. **3D Proof-Object for CT Reports.** We formalize 3D chest CT report generation as producing *(report text + explicit citations to deterministic 3D spatial tokens)*, enabling deterministic auditing of each finding.
2. **Budgeted Evidence Acquisition (Inference-Time Closed Loop).** We introduce ProveTok-Agent, an inference-time closed loop that alternates *write → verify → acquire (split) → rewrite*, producing a **reliability–budget Pareto curve**. We compare a heuristic split rule with a learned split policy trained via DPO on verifier-derived preferences.

We keep RL-style long-horizon policy optimization out of the main claim (Appendix / future work), aligning with reviewer expectations on scope and reproducibility.

---

## 2. Related Work

### 2.1 3D CT report generation

CT2Rep introduced an early dedicated approach for generating radiology reports from 3D medical imaging volumes and established baselines on chest CT.([MICCAI Papers][1]) Subsequent work increasingly integrates LLMs: Dia-LLaMA adapts LLaMA2-7B for chest CT reporting using disease-aware attention, prototype memory, and diagnostic prompts, evaluated on a CTRG-Chest-548K setting that (in the reproducible subset) contains 1,804 CT-report pairs.([MICCAI Papers][4]) MvKeTR proposes multi-view perception and retrieval-based knowledge enhancement for chest CT report generation.([arXiv][7]) 3D VLM lines (e.g., 3D-CT-GPT++, μ²Tokenizer/μ²LLM) further leverage multimodal LLM architectures and preference optimization guided by evaluation metrics such as GREEN.([OpenReview][8])

**Gap:** Across these systems, evidence grounding is usually implicit (attention maps) or post-hoc, rather than an explicit, deterministic, auditable 3D proof object attached to each finding.

### 2.2 Datasets enabling 3D CT report learning and grounding

CT-RATE is a large public dataset pairing 3D chest CT volumes with free-text reports, consisting of 25,692 non-contrast CT scans from 21,304 patients and expanded reconstructions; it also enables CT-CLIP and CT-CHAT.([arXiv][9])
RadGenome-Chest CT extends CT-RATE with segmentation masks and grounded report sentences, providing 197 organ masks, 665K grounded report sentences, and 1.2M grounded VQA pairs.([Nature][2]) This makes RadGenome uniquely suitable for **quantitative evaluation of citation grounding** in 3D CT reporting.

### 2.3 Clinical alignment and evaluation metrics

Clinical report metrics have evolved beyond surface NLG overlap. CheXbert provides label extraction for report-level clinical efficacy.([ACL Anthology][5]) RadGraph extracts entities and relations for structure-aware scoring.([arXiv][10]) RadCliQ was proposed as a clinically correlated composite evaluation for report generation.([Cell][11]) GREEN uses LLM-based evaluation with radiology-specific error notions.([ACL Anthology][6]) RaTEScore targets radiology report generation evaluation from another angle.([ACL Anthology][12])

On optimization, RRG-DPO shows direct preference optimization can improve clinical metrics on both 2D and 3D datasets (including CT-RATE), but does not enforce per-finding 3D evidence traceability.([MICCAI Papers][3])

---

## 3. Task Definition: Proof-Carrying CT Report Generation Under a Budget

### 3.1 Inputs

Given:

* A preprocessed 3D chest CT volume (V \in \mathbb{R}^{H \times W \times D}).
* A token budget (B) (maximum number of spatial evidence tokens allowed at any step).
* An octree refinement limit (L_{\max}) and maximum interaction steps (T_{\max}).

### 3.2 Deterministic spatial tokenization (ProveTok)

Let (\mathcal{C}_t) be a set of 3D axis-aligned cells forming an octree partition at step (t).
A deterministic tokenizer ( \textsc{Tokenize}(V, \mathcal{C}_t) \rightarrow \mathcal{T}_t ) produces a set of tokens:
[
\mathcal{T}_t = { \tau_i = (id_i, \ell_i, b_i, e_i, p_i) }
]
where:

* (id_i): stable token id (derived from octree path / Morton code).
* (\ell_i): octree level.
* (b_i): cell bounds in voxel and mm coordinates.
* (e_i): token embedding (ROI pooled from a 3D backbone).
* (p_i): optional auxiliary attributes (e.g., uncertainty; not required for hard verification in v1.1).

Splitting a cell (c) replaces it with its 8 children. Token count increases by (+7).

### 3.3 Outputs: report + proof-object

The model outputs:

1. **Findings**: a list of lines ({f_k}). Each line must end with an explicit citation list:

   * Example:
     `Right pleural effusion. [CITE: 17, 23]`
2. **Impression**: optional summary constrained to not introduce new evidence ids (only reuse cited ids from Findings).
3. **Proof-object**: a mapping from finding index (k) to cited token ids (S_k \subseteq {id_i}).

This produces an **auditable artifact**: every statement is linked to explicit 3D regions via token bounds.

---

## 4. ProveTok-Agent

### 4.1 Overview

ProveTok-Agent is a single closed-loop agent that repeatedly:

1. writes a finding with citations,
2. runs a deterministic verifier,
3. if issues remain and budget allows, acquires finer evidence by splitting tokens,
4. rewrites the affected finding.

**Figure 1 (place in Introduction or Method Overview).**
Diagram: CT volume → ProveTok tokens (octree) → LLM writes finding lines with `[CITE]` → Verifier outputs issues + blamed token ids → Split policy selects token to split → loop; plus a bottom plot “issue vs budget curve”.

### 4.2 Visual encoder and token embeddings

We employ a 3D vision backbone (E_{\text{3D}}) to compute a dense feature volume (F = E_{\text{3D}}(V)). A 3D ROI pooling operator extracts token embeddings:
[
e_i = \textsc{ROIPool}(F, b_i) \in \mathbb{R}^{d}
]
We then add lightweight positional metadata (level embedding + normalized center coordinates) and project into the LLM hidden size with an MLP.

**Implementation choice (write explicitly for reproducibility):**

* Backbone options (report both in appendix; pick one as main):
  (i) CT-CLIP encoder initialized from CT-RATE pretraining; CT-RATE/CT-CLIP/CT-CHAT are described in the CT-RATE foundation model work.([arXiv][9])
  (ii) RadFM as a general 2D&3D radiology foundation model (strong baseline).([Nature][13])
* We recommend using CT-CLIP for main experiments due to tight domain match and open tooling, and reporting RadFM swap-in as an ablation.

### 4.3 Proof-carrying generation with a cite pointer

The generator is an LLM (G_\theta) conditioned on token embeddings (\mathcal{T}_t). It emits one finding line at a time (for stability and short verifier loops).

We enforce a strict decoding grammar:

* Each finding line **must** end with `[CITE: i1, i2, ...]`.
* Citation ids must be valid token ids present in (\mathcal{T}_t).
* Maximum citations per line (K_{\max}) (default 8).

To generate citations, we attach a pointer head:
[
\pi_\theta(id \mid h_{\text{cite}}, {e_i}) = \textsc{Softmax}( e_i^\top W h_{\text{cite}} )
]
and decode a small set of ids (top-k or sampled without replacement).

**Training signal for citations (main paper, no RL):**

* On RadGenome, we have sentence↔region grounding; we map each grounded region mask to the overlapping ProveTok cells, giving supervised citation sets.([Nature][2])
* On CT-RATE-only cases, we train text generation normally and optionally use weak citation supervision (Appendix).

### 4.4 Hard verifier (deterministic, auditable)

The verifier takes a finding line (f_k) and its citations (S_k), and deterministically emits issues.

We define four **hard** rules (v1.1 scope):

**R1. NoCitation.** Missing `[CITE: ...]` or empty citation list.

**R2. CoarseOnly.** If a finding asserts a positive abnormality (detected via lexicon of “present/seen/effusion/nodule/…”), then at least one cited token must have level (\ell \ge \ell_{\min}) (default (\ell_{\min}=2)). If not, issue = CoarseOnly.

**R3. LateralityMismatch.** If the text specifies “left” (resp. “right”), then all cited tokens must lie on the left (resp. right) side of the mid-sagittal plane, computed deterministically from standardized volume coordinates.

**R4. BilateralSeparation.** If the text specifies “bilateral” or mentions both sides, then citations must include at least one left-side token and one right-side token.

Each issue also returns a **blame set** of token ids involved (e.g., tokens violating laterality, or all cited tokens if coarse-only).

> **Why this verifier is reviewer-safe:** it depends only on (i) cited ids and (ii) token bounds and (iii) lexicon parsing. It is deterministic and fully auditable—no learned judge needed in the main claim.

### 4.5 Budgeted evidence acquisition loop

At inference, ProveTok-Agent executes:

```text
Initialize octree cells at level 1 (8 tokens)
for t = 1..T_max:
  tokens = Tokenize(V, cells)
  write next finding line with citations
  issues = Verify(finding, tokens)
  if issues empty: accept and move to next finding
  else if |tokens| >= B or no splittable blamed token: rewrite by de-specifying (uncertain) and accept
  else: choose a blamed token and SPLIT it (octree refinement), then rewrite the same finding
Return full report + proof-object + logs
```

**Figure 2 (place in Method).**
Show an example of a coarse token and its 8 children, with ids and bounds, illustrating the +7 budget accounting and why splitting increases spatial precision.

---

## 5. Split Policy: Heuristic vs Learned (DPO)

### 5.1 Heuristic split policy (strong deterministic baseline)

Given a set of issues for the current finding, we collect blamed token ids (B_t) and select:
[
id^* = \arg\max_{id \in B_t} ; \textsc{Priority}(\text{issue}, id)
]
Default priority:

* Prefer tokens with smaller level (coarser) if CoarseOnly triggered (splitting coarse tokens yields maximal gain).
* Prefer tokens violating laterality boundary if LateralityMismatch triggered (refine near boundary).
* Tie-break by token volume (split the largest blamed cell).

This heuristic is simple, deterministic, and reproducible.

### 5.2 Learned split policy via verifier-derived DPO (main paper “learned”)

We train a policy (\pi_\theta(id \mid s_t)) over splittable token ids, where state (s_t) includes:

* issue types and severities (hard-coded),
* cited token ids and their geometry (level, bounds, center),
* remaining budget and current step.

**Preference data construction (offline, one-step lookahead):**
For each state (s_t), sample (N) candidate split actions (id_1,\dots,id_N). For each action:

1. apply split,
2. rewrite the current finding (same LLM),
3. re-run verifier, obtaining (\Delta \textsc{Issue}).

Define preferred action (id^+) as the one maximizing issue reduction (tie-break by lower token cost / fewer citations). This creates pairs ((s_t, id^+, id^-)).

**DPO objective (discrete-action variant):**
Let (\pi_0) be the reference policy (heuristic or an IL warm-start). We optimize:
[
\mathcal{L}*{\text{DPO}}(\theta) = -\mathbb{E}\Big[\log \sigma\big(\beta((\log \pi*\theta(id^+|s)-\log \pi_\theta(id^-|s))-(\log \pi_0(id^+|s)-\log \pi_0(id^-|s)))\big)\Big]
]
This keeps training stable and avoids full RL in the main paper, while still learning an improved acquisition strategy.

**Figure 3 (place in Experiments).**
Show the “preference generation” pipeline: a state → multiple candidate splits → one-step issue reduction → preference pair → DPO update.

---

## 6. Experiments

### 6.1 Datasets and splits

**RadGenome-Chest CT (main).**
Built on CT-RATE, RadGenome provides:

* 197 organ-level segmentation categories,
* 665K grounded report sentences linked to segmentation masks,
* 1.2M grounded VQA pairs.([Nature][2])
  This enables direct evaluation of whether cited ProveTok regions overlap the ground-truth grounded regions.

**CT-RATE (secondary).**
CT-RATE pairs 3D chest CT volumes with reports and multi-abnormality labels; the foundation model paper describes CT-RATE as 25,692 scans from 21,304 patients (expanded reconstructions).([arXiv][9])
For comparability with prior 3D report-generation literature, we adopt the preprocessing described in RRG-DPO (§6.2) and report development results on a fixed held-out patient-level test split (the default in this repo). **SOTA** claims on CT-RATE are only made when additionally reporting on the official RRG-DPO split: training (24,128 volumes / 20,000 patients) and test split into validation (360/300) and testing (1204/1004).([MICCAI Papers][3])

**CTRG-Chest-548K subset (appendix).**
We include the commonly reproduced 1,804-pair setting described in Dia-LLaMA (80/20 split).([MICCAI Papers][4])

### 6.2 Preprocessing

We standardize voxel spacing and volume size following prior work for CT-RATE comparability. In particular, RRG-DPO reports resampling to 0.75×0.75×1.5 mm³ and center-cropping/padding to 480×480×240 voxels; we adopt this pipeline for CT-RATE experiments.([MICCAI Papers][3])
For RadGenome, we follow its provided standardized volumes/masks (and report any resampling in Appendix if we unify pipelines).([Nature][2])

### 6.3 Compared methods (baselines)

#### External report-generation baselines (text-only evaluation)

We include representative 3D CT report generators and 3D VLM variants:

* **CT2Rep** (3D CT report generation baseline).([MICCAI Papers][1])
* **Dia-LLaMA** (LLM-driven CTRG with diagnostic prompts; 1,804-pair setting).([MICCAI Papers][4])
* **MvKeTR** (multi-view + retrieval knowledge enhancement).([arXiv][7])
* **3D-CT-GPT++** (3D encoder optimized; evaluated with GREEN among others).([OpenReview][8])
* **μ²Tokenizer/μ²LLM** (multi-scale tokenizer + DPO guided by GREEN).([arXiv][14])

> 注意：这些 baseline 大多不输出 citations。因此我们在 proof 指标上给它们一个 **post-hoc citation wrapper**（下节），以保证公平。

**Reproducible evaluation protocol (what we actually run).**
We do not need to integrate each baseline’s training/inference code. Instead, each external baseline provides a unified `pred_jsonl`:
`{sample_id=<scan_hash>, method, pred_text}`. We join it with references from a manifest to build `pairs_all.jsonl`, then compute Table1 metrics + paired statistics via a single driver (`scripts/paper/run_table1_ct_rate.py` → `scripts/paper/compute_paper_metrics.py`). Optional metrics (GREEN/RadCliQ/CheXbert) are exported as `extra_metrics_jsonl` via RadEval (Py3.11) and merged post-hoc; optional proof metrics for non-proof baselines are computed with the post-hoc citation wrapper (audit-only; does not change text). For any “SOTA” claim, we require **100% coverage** of the target test split for every compared method.

#### Our system variants (核心消融)

* **NoProof**: standard report generation without citations (upper bound on hallucination risk).
* **Proof-NoVerify**: citations required, but no verifier loop (tests whether “proof contract” alone helps).
* **Proof+Verify-NoSplit**: verifier runs, but never splits; only rewrites (tests necessity of acquisition).
* **Proof+Verify+Split-Heuristic**: full loop with heuristic split.
* **Proof+Verify+Split-DPO**: full loop with learned split policy (ours).

### 6.4 Metrics

#### Standard RRG metrics (comparability)

* **NLG**: BLEU-4, METEOR, ROUGE-L (standard).
* **Clinical efficacy**: CheXbert precision/recall/F1 (label extraction).([ACL Anthology][5])
* **Structured**: RadGraph F1.([arXiv][10])
* **Clinical correlation**: RadCliQ.([Cell][11])
* **LLM-based**: GREEN.([ACL Anthology][6])
* **Additional**: RaTEScore.([ACL Anthology][12])

Implementation note: we export heavy/LLM-based metrics (GREEN/RadCliQ/CheXbert) as per-sample `extra_metrics_jsonl` in an isolated environment (RadEval, Py3.11) and merge them into the same Table1 computation, to pin versions and avoid hidden dependency drift.

#### Proof-centric reliability metrics (核心主指标)

We report:

* **NoCitation rate** (R1),
* **CoarseOnly rate** (R2),
* **LateralityMismatch rate** (R3),
* **BilateralSeparation violations** (R4),
* A weighted summary:
  [
  \textsc{WeightedIssue} = w_1 R1 + w_2 R2 + w_3 R3 + w_4 R4
  ]
  Default weights: (w_1=3) (missing citation is fatal), (w_2=w_3=w_4=2).

We plot **reliability–budget Pareto curves**: WeightedIssue vs token budget (B).

#### Evidence grounding metrics (RadGenome only)

Using grounded sentence↔mask annotations:

* **Hit@M**: fraction of cited tokens whose cells overlap the grounded mask (IoU>0).
* **Coverage**: fraction of grounded mask voxels covered by union of cited token cells.
* **Laterality grounding accuracy**: for left/right sentences, fraction whose cited tokens lie exclusively on correct side.

### 6.5 Post-hoc citation wrapper (fairness for non-proof baselines)

For baselines that do not emit citations, we attach citations as follows:

* Parse each generated finding line (same lexicon as verifier).
* Select top-(K) tokens from current (\mathcal{T}_t) by a deterministic rule (e.g., highest overlap with predicted anatomy region; or highest token saliency if available).
* Attach `[CITE: ...]` for auditing only (does not change the baseline’s text).

This ensures proof metrics compare “ability to produce auditably supported statements” rather than “format compliance”.

### 6.6 Implementation details (write this verbatim in paper)

* Tokenization: octree, init_level=1, max_level=3, budgets {64,128,256,512}, max_steps=32.
* Decoding: finding-by-finding; max citations per finding (K_{\max}=8).
* LLM: 7B class with LoRA (for parity with Dia-LLaMA’s typical setup).([MICCAI Papers][4])
* Statistical tests: bootstrap CI over studies; Wilcoxon signed-rank for paired comparisons, following prior evaluation practice in RRG-DPO.([MICCAI Papers][3])

---

## 7. Results (表格与图：给你“可直接替换数字”的终稿模板)

> 我不能在这里“编造”你方法的数值结果；但我会把主文中 reviewer 期待看到的每一张表/图 **完整写成可直接粘贴的终稿结构**，你跑完实验只要填数字，不需要再改叙事。

### 7.1 Main reliability–budget Pareto (核心图)

**Figure 4 (Main).** Plot WeightedIssue vs budget (B) for:

* Proof+Verify-NoSplit,
* +Split-Heuristic,
* +Split-DPO (ours),
  plus optionally a dashed line for Proof-NoVerify.

**Expected qualitative claim (写法，不依赖具体数值)：**

* The curve for acquisition dominates no-acquisition across budgets (strictly lower issues).
* DPO policy dominates heuristic at medium budgets (128–256) where selection matters most.
* At very low budgets (64), all methods converge (insufficient capacity); at very high budgets (512), diminishing returns appear.

> 这段写法 reviewer 很难挑，因为它不承诺具体数值，只承诺“曲线支配关系”，并且你只要训练正常就很容易成立。

### 7.2 Quantitative table: standard metrics + proof metrics

**Table 1 (CT-RATE test).**
Columns: BLEU-4 / METEOR / ROUGE-L / CheXbert F1 / RadGraph / RadCliQ / GREEN / RaTEScore / WeightedIssue@B=256.
Rows:

* CT2Rep([MICCAI Papers][1])
* Dia-LLaMA([MICCAI Papers][4])
* MvKeTR([arXiv][7])
* 3D-CT-GPT++([OpenReview][8])
* μ²Tokenizer([arXiv][14])
* Proof+Verify-NoSplit (ours ablation)
* +Split-Heuristic (ours)
* +Split-DPO (ours, main)

**写作要点（放表后两段，直接可用）：**

* “While several baselines improve textual/clinical metrics, they do not optimize for auditable evidence. Our proof-centric loop yields consistently lower verifier issues under fixed budgets, without degrading clinical efficacy metrics.”
* “Post-hoc citation wrappers do not close the gap, suggesting that evidence acquisition must be integrated into the generation loop rather than appended after generation.”

### 7.3 Grounding table (RadGenome)

**Table 2 (RadGenome validation/test).**
Columns: Hit@8 / Coverage@8 / Laterality-grounding-acc / WeightedIssue@B=128 and @B=256.
Rows: same as above; baselines use wrapper citations.

**写作要点：**

* 强调：RadGenome 提供 grounded sentence↔mask，因此 citation quality 可定量评估。([Nature][2])
* 你的系统若真的 work，应该在 Hit/Coverage 上显著更高，且 laterality grounding 更稳定。

### 7.4 Ablation table (你两大贡献逐条“不可替代”)

**Table 3 (Ablations).**
Rows:

* NoProof
* Proof-NoVerify
* Proof+Verify-NoSplit
* Proof+Verify+Split-Heuristic
* Proof+Verify+Split-DPO

Columns:

* WeightedIssue@B=128/256
* LateralityMismatch rate
* CoarseOnly rate
* CheXbert F1 (防止“全 uncertain”投机)([ACL Anthology][5])

**写作要点：**

* Proof contract alone reduces NoCitation but not CoarseOnly/Laterality.
* Verifier without split shifts errors from hallucination to hedged language (showing the need for acquisition).
* Split reduces coarse-only and laterality mismatch at same budget; learned policy further reduces residual issues.

### 7.5 Qualitative case study (不可缺，MICCAI 很吃)

**Figure 5.** One CT case with:

* generated findings lines,
* citations,
* visual overlay of cited cells on axial slices (at least 2 slices: one coarse step, one after split),
* verifier issues before/after.

**写作要点：**

* show a laterality error resolved by splitting boundary-adjacent cell and re-citing correct side.
* show an over-specific claim downgraded when budget exhausted.

---

## 8. Discussion

### 8.1 Why proof-objects matter in 3D CT

Unlike report-level preference optimization (e.g., RRG-DPO), proof-objects attach an explicit spatial audit trail to each finding, enabling deterministic detection of laterality and granularity failures that are common in volumetric reasoning.([MICCAI Papers][3])

### 8.2 Budgeted acquisition as a first-class objective

Many recent models implicitly increase compute by denser tokenization (e.g., multi-scale tokenizers) or larger backbones. Our approach instead exposes a controllable knob—**evidence budget**—and reports reliability as a function of budget, yielding a more deployment-relevant characterization.([arXiv][14])

### 8.3 Limitations (写得越狠越加分)

1. **Verifier coverage is incomplete.** We only enforce four hard rules; subtle clinical contradictions may remain. (We intentionally keep verifier deterministic and minimal in the main claim.)
2. **Lexicon-based abnormality detection is imperfect.** We mitigate by restricting hard rules to laterality/granularity/citation structure; richer NLP parsing is future work.
3. **Dataset bias and generalization.** CT-RATE/RadGenome are non-contrast chest CT; generalizing to contrast phases or other anatomies needs validation.([arXiv][9])
4. **Inference cost.** Repeated verify/split loops add overhead; we bound runtime by (T_{\max}) and show Pareto curves.

### 8.4 Future work (把 RL 放这里/附录，与你要求一致)

We see two natural extensions: (i) soft preference signals and richer verifier coverage, and (ii) long-horizon optimization (e.g., GRPO/PPO) using clinically correlated rewards. We omit these from the main claim for scope and reproducibility.

---

## 9. Conclusion

We introduced ProveTok-Agent, a proof-carrying and budgeted evidence acquisition paradigm for 3D chest CT report generation. By enforcing explicit citations to deterministic 3D tokens and closing the loop with a hard verifier and controlled token refinement, our method enables reliable, auditable report generation and exposes a deployment-relevant reliability–budget trade-off. Learned acquisition via DPO further improves token refinement decisions beyond deterministic heuristics.

---

## Ethics & Clinical Use Statement (MICCAI 友好写法)

This work is for research use only and is not intended for direct clinical deployment without prospective evaluation. All models may produce errors; the explicit proof-object is designed to support auditing rather than replace clinician judgment.

---

# Appendix (写在主文之外，但你可以现在就准备）

* RL/GRPO 的设定与 reward 设计（放这里）。
* LowScore/uncertainty 的训练与校准（放这里）。
* CTRG-548K(1,804) 的对齐对比实验（放这里）。([MICCAI Papers][4])

---

## B. 你现在就可以粘到论文里的 “Tables/Figures LaTeX skeleton”

> 下面是“无数字版本”，你跑完实验填数即可。**注意：我这里不放任何我系统的引用标号**，避免你复制到 LaTeX 时要清理。

```latex
% Table 1: CT-RATE
\begin{table}[t]
\centering
\caption{CT-RATE test set results. We report standard RRG metrics and proof-centric reliability at budget $B=256$.}
\begin{tabular}{lcccccccc}
\toprule
Method & BLEU4$\uparrow$ & METEOR$\uparrow$ & ROUGE-L$\uparrow$ & CheXbert F1$\uparrow$ & RadGraph$\uparrow$ & RadCliQ$\downarrow$ & GREEN$\uparrow$ & WeightedIssue$\downarrow$ \\
\midrule
CT2Rep \\
Dia-LLaMA \\
MvKeTR \\
3D-CT-GPT++ \\
$\mu^2$Tokenizer \\
\midrule
Ours: Proof+Verify-NoSplit \\
Ours: +Split-Heuristic \\
Ours: +Split-DPO \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
% Figure: Pareto curve
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{pareto_issue_budget.pdf}
\caption{Reliability--budget Pareto curves on CT-RATE/RadGenome. Lower is better.}
\end{figure}
```

---

## C. 参考文献（给你 BibTeX，基于联网结果可直接补齐）

> 你在 Overleaf 里把这些条目放进 `.bib`，然后按你文中引用 key 使用即可。

```bibtex
@article{hamamci2024ctrate,
  title={Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and others},
  journal={arXiv:2403.17834},
  year={2024}
}

@inproceedings{hamamci2024ct2rep,
  title={CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Menze, Bjoern},
  booktitle={MICCAI},
  year={2024}
}

@article{zhang2025radgenome,
  title={Development of a large-scale grounded vision language dataset for chest CT analysis},
  author={Zhang, Xiaoman and Wu, Chaoyi and Zhao, Ziheng and Zhang, Ya and Xie, Weidi and Wang, Yanfeng and others},
  journal={Scientific Data},
  year={2025}
}

@inproceedings{chen2025diallama,
  title={Dia-LLaMA: Towards Large Language Model-driven CT Report Generation},
  author={Chen, Z. and others},
  booktitle={MICCAI},
  year={2025}
}

@article{liu2025rrgdpo,
  title={RRG-DPO: Direct Preference Optimization for Radiology Report Generation},
  author={Liu, H. and others},
  booktitle={MICCAI},
  year={2025}
}

@article{yu2023radcliq,
  title={Evaluating progress in automatic chest X-ray radiology report generation},
  author={Yu, Feiyang and Endo, Mark and others},
  journal={Patterns},
  year={2023}
}

@article{jain2021radgraph,
  title={RadGraph: Extracting Clinical Entities and Relations from Radiology Reports},
  author={Jain, Saahil and others},
  year={2021}
}

@article{grace2024green,
  title={Generative Radiology Report Evaluation and Error Notation (GREEN)},
  author={Delbrouck, Jean-Benoit and others},
  year={2024}
}

@article{li2025mu2tokenizer,
  title={${\mu}^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation},
  author={Li, Siyou and others},
  journal={arXiv:2507.00316},
  year={2025}
}

@article{deng2024mvketr,
  title={MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancement},
  author={Deng, Xiwei and others},
  journal={arXiv:2411.18309},
  year={2024}
}
```

> 注：arXiv / MICCAI / Nature 这几条的作者列表你最终建议用官方 BibTeX 自动导出替换 `and others`，但主文叙事已经足够严谨。

---

# D. Reviewer “毒舌预演”（你写主文时要提前堵枪眼）

下面是我按 MICCAI reviewer 口吻，最可能攻击你的点，以及你在 v1.1 主文里应该如何“一句话封死”：

1. **“你这不就是把 attention/patch 叫 token，强行说是证据？”**

* 回击要点：我们 token **有确定性 3D bounds + 稳定 id + 显式引用语法 + deterministic verifier**，这和 attention heatmap 根本不是一类产物（attention 不可审计/不可复现）。

2. **“verifier 规则太简单，临床事实性没保证。”**

* 回击要点：我们主文刻意只承诺 **laterality+granularity+citation contract** 的 hard safety（这些是 3D 最常见、可确定性审计的问题），并用 RadGenome grounding 指标证明 citations 不是摆设。([Nature][2])

3. **“learned split 可能只是 overfit / 或者不如把 token 直接变多。”**

* 回击要点：我们给出 **Pareto 曲线**，在同 budget 下比较；并给出 Heuristic vs DPO 的消融，证明“会挑着 split”在中等预算最关键。

4. **“你没有跟 SOTA 3D VLM 比。”**

* 你主文表 1/2 列出 CT2Rep、Dia-LLaMA、MvKeTR、3D-CT-GPT++、μ²Tokenizer；这些都可在 related work 里据实引用。([MICCAI Papers][1])
* proof 指标对这些 baseline 用 wrapper，保证公平。

---

如果你下一步要我继续把它“变成真正 Overleaf 可直接编译的 8 页 LaTeX（含 Algorithm 环境、图注、表注、附录、参考文献引用 key 全对齐）”，我也可以直接给你一份 **完整 LaTeX 工程骨架**（main.tex + sections + bib），你只需要把实验数字填进去即可。

[1]: https://papers.miccai.org/miccai-2024/paper/2185_paper.pdf "https://papers.miccai.org/miccai-2024/paper/2185_paper.pdf"
[2]: https://www.nature.com/articles/s41597-025-05922-9 "https://www.nature.com/articles/s41597-025-05922-9"
[3]: https://papers.miccai.org/miccai-2025/paper/1273_paper.pdf "https://papers.miccai.org/miccai-2025/paper/1273_paper.pdf"
[4]: https://papers.miccai.org/miccai-2025/paper/3319_paper.pdf "https://papers.miccai.org/miccai-2025/paper/3319_paper.pdf"
[5]: https://aclanthology.org/2020.emnlp-main.117/ "https://aclanthology.org/2020.emnlp-main.117/"
[6]: https://aclanthology.org/2024.findings-emnlp.21.pdf "https://aclanthology.org/2024.findings-emnlp.21.pdf"
[7]: https://arxiv.org/abs/2411.18309 "https://arxiv.org/abs/2411.18309"
[8]: https://openreview.net/forum?id=LzycEbgLoi "https://openreview.net/forum?id=LzycEbgLoi"
[9]: https://arxiv.org/abs/2403.17834 "https://arxiv.org/abs/2403.17834"
[10]: https://arxiv.org/pdf/2106.14463?utm_source=chatgpt.com "Extracting Clinical Entities and Relations from Radiology ..."
[11]: https://www.cell.com/patterns/pdf/S2666-3899%2823%2900157-5.pdf "https://www.cell.com/patterns/pdf/S2666-3899%2823%2900157-5.pdf"
[12]: https://aclanthology.org/2024.emnlp-main.836/ "https://aclanthology.org/2024.emnlp-main.836/"
[13]: https://www.nature.com/articles/s41467-025-62385-7 "https://www.nature.com/articles/s41467-025-62385-7"
[14]: https://arxiv.org/abs/2507.00316 "https://arxiv.org/abs/2507.00316"
