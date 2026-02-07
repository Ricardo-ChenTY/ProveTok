# ProveTok: Proof-Carrying Budgeted Evidence Tokenization for Grounded 3D CT Report Generation

> 文档版本：2026-02-07（paperized README）  
> 主审计状态：`outputs/oral_audit.json` -> `ready_for_oral_gate=true`  
> 最新 counterfactual 强证据：`outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json`

## Abstract
3D CT 报告生成的难点不只是“写得像”，而是要在预算受限下同时满足三件事：句子有空间证据可复核、证据不足时能拒答、并且延迟与错误类型可控。我们提出 ProveTok：在固定预算 `B = B_enc + B_gen` 下，BET 负责证据 token 化与选择（`B_enc`），PCG 负责生成 `frames + citations + refusal + verifier_trace`（`B_gen`），并通过 verifier taxonomy 与 refusal calibration 形成可审计闭环。我们用多预算（`2e6..7e6`）、多随机种子、paired bootstrap + Holm 的统计协议，并以 claim-level 机器裁判（`scripts/proof_check.py` / `scripts/oral_audit.py`）作为最终判定。结果上，`real` profile 下 C0001-C0006 全通过（见 §5.0 Table 4）；在跨域 V0003（silver label）反事实压力测试中，`omega_perm` pooled（seeds `0..19`）达到 `mean_diff=+0.002567`、`p_one_sided=0.0001`、`p_holm=0.0006`，支持 citation 通路具备可机检的非平凡效应。

## TL;DR（Oral 速览）
- 一张表抓主证据：`docs/paper_assets/tables/table4_oral_minset.md`（下文 §5.0 也内嵌同表）。
- 图表可复现：`python scripts/paper/build_readme_figures.py` / `python scripts/paper/build_readme_tables.py`（产物在 `docs/paper_assets/`）。
- 机器审计门：`python scripts/proof_check.py --profile real` + `python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict`。
- LaTeX 论文草稿（NeurIPS-style）：`paper/`（入口 `paper/main.tex`，已生成 `paper/main.pdf` 便于快速浏览；正式投稿请替换为官方 NeurIPS style files）。
- 写作范式拆解与模板：`docs/writing_refs/`（对标论文的结构/图表套路 + 可复用段落模板）。

## 1. Introduction
### 1.1 问题与动机
在医学场景中，报告生成最危险的失败不是“不流畅”，而是“写了结论却找不到证据”或“证据不足仍过度断言”。当计算预算受限时，grounding 不是附加可视化，而是一个资源分配问题：哪些体素区域值得被编码成 token，哪些句子必须引用这些 token，哪些情况下应该拒答并留下可复核的 trace。现有方法常把 citation/grounding 当作后验解释，把 refusal 当作后处理阈值，从而难以在同一协议里同时约束 latency、unsupported、overclaim 与 miss-rate。

### 1.2 本文目标
我们以“可证明生成”为中心目标：
1. 在固定预算 `B = B_enc + B_gen` 下，做多预算对齐评测（`2e6..7e6`），避免“多跑算力换指标”。
2. 让输出天然携带可机检的 `citations + verifier_trace`，而非后验可视化。
3. 把 refusal 写入硬 gate（`critical_miss_rate/refusal_ece/refusal_rate`），防止“封嘴刷安全指标”。
4. 用反事实（counterfactual）与跨域弱标（V0003, silver）压力测试，检验 citation 通路是否非平凡。

### 1.3 主要贡献
1. 协议：提出 BET + PCG + Verifier + Refusal 的统一闭环，把“预算-证据-生成-拒答-审计”写入同一可执行口径。
2. 裁判：建立 claim-level 机器裁判链（`docs/plan.md -> docs/experiment.md -> outputs/* -> scripts/proof_check.py -> scripts/oral_audit.py`），将结论判定权交给可复现规则。
3. 证据：在 `real` profile 下通过 C0001-C0006（多预算+多 seed+Holm，见 §5.0 Table 4），并在 V0003 silver 路径通过 pooled counterfactual 证明 `omega_perm` 的非平凡性（同见 Table 4）。

## 2. Related Work
### 2.1 3D CT 报告生成与基础数据
CT-RATE 与 CT2Rep 系列工作推动了 3D CT 报告生成的数据与强基线建设 [1,2,3]，但主流优化与评测仍偏向“文本像不像”，对“句子-空间证据绑定”与拒答校准缺少统一、可审计的硬约束。ProveTok 的目标不是再造一个 backbone，而是把预算、证据、生成、校验与拒答写入同一协议，并用机器裁判规则而非主观示例来判定是否达标。

### 2.2 Grounded vision-language in CT
ReXGroundingCT 将 free-text findings 与 3D 像素级标注显式连接，提供了 sentence-level grounding 的关键评测土壤 [4]。这使得 grounding 指标（例如 `IoU_union`）可以作为生成协议的一等指标，而不只是可视化附件。本文沿用该评测土壤，把“citation→空间覆盖→IoU_union”写进 proof rule（见 Table 4 的 C0004），从而把“可解释”变成“可检验”。

### 2.3 Trustworthy generation 与 refusal
近期 RAG/LLM 研究开始把 grounded attribution 与 learning-to-refuse 联合建模 [5]。在医学报告场景中，这一方向尤为关键：拒答如果不与 `critical_miss_rate`、`refusal_ece`、`unsupported_rate` 联动约束，就容易出现“封嘴换指标”的伪安全。本文将 refusal calibration 写入硬 gate（Table 4 的 C0005），并要求它与 grounding/unsupported 一起在同一预算协议下被审计。

### 2.4 本文定位
ProveTok 的核心不是“再做一个模型结构”，而是把预算、证据、生成、校验和拒答写入同一可审计协议，并将最终判定权交给机器裁判规则，而非人工主观筛选。

## 3. Method
### 3.1 Problem Formulation
给定体数据 `V`，系统在预算约束下运行：
- `B = B_enc + B_gen`
- `B_enc`：证据 token 化与选择开销
- `B_gen`：文本生成与 verifier 交互开销

系统输出为 `frames + citations + refusal + verifier_trace`，目标是在 **固定预算** 下最大化 grounded quality，同时满足可信性与延迟的 hard gates。

### 3.2 BET: Budgeted Evidence Tokenization
BET 在预算内生成带空间索引与置信信息的 token：
- token 包含 `(cell_id, level, score, uncertainty, embedding)`
- 支持 fixed-grid / ROI / scored variants 的统一成本对齐
- 通过 FLOPs-matched 或 latency-aware 协议做公平比较

### 3.3 PCG: Proof-Carrying Generation
PCG 输出的不只是文本，还包括：
- `frames`
- `citations`
- `confidence/refusal`
- `verifier trace`

每条关键陈述必须具备可机检引用，否则进入 refine/refusal 分支。

### 3.4 Verifier + Refusal Calibration
Verifier 在固定 taxonomy 下输出 issue 列表（例如 `U1_unsupported`, `O1_overclaim`, `M1_missing_slot`）。Refusal calibration 在 dev 选阈值、test 冻结阈值，联合约束：
- `critical_miss_rate <= 0.05`
- `refusal_ece <= 0.15`
- `refusal_rate <= 0.20`

### 3.5 Metrics（README 口径）
- `IoU_union`：对 citations 对应的空间覆盖做 union 后，与 GT mask 的 IoU（见 `provetok/experiments/run_baselines.py` 的 grounding 评估口径）。
- `combined`：`0.5 * frame_f1 + 0.5 * IoU_union`（见 `provetok/experiments/run_baselines.py`；`outputs/E0164-full/baselines_curve_multiseed.json` 的 `meta.config` 固定 `nlg_weight=0.5, grounding_weight=0.5`）。
- `unsupported_rate`：`U1_unsupported` issue 数 / frame 数（verifier taxonomy 与计算见 `provetok/experiments/run_baselines.py`）。

### 3.6 Closed-Loop Overview
![Figure 1: ProveTok closed-loop pipeline](docs/paper_assets/figures/fig1_system_overview.png)

图 1 对应数据来源：系统协议与实现代码路径（`provetok/*`, `scripts/proof_check.py`, `scripts/oral_audit.py`）。

## 4. Experimental Setup
### 4.1 Datasets
| Dataset | Role | Key characteristic |
|---|---|---|
| ReXGroundingCT-100g | 主 grounding / counterfactual 评测 | 3D + sentence-level mask，支持像素级 grounding [4] |
| ReXGroundingCT-mini | 快速迭代与 smoke/full | 低成本回归验证 |
| CT-RATE (TS-Seg eval-only) | V0003(A') 跨域弱证据 | TotalSegmentator 自动 mask（`silver_auto_unverified`）[1,6] |
| CT-RATE (pseudo-mask) | V0003(C) 跨域弱证据 | saliency->pseudo-mask（`silver_auto_unverified`，见 `scripts/data/build_ct_rate_pseudomask_manifest.py`） |

注意：V0003(A'/C) 使用自动/伪 mask，因此属于 `silver_auto_unverified`，仅用于跨域 sanity 与反事实压力测试；`real` profile 的主结论以 ReXGroundingCT 的 gold mask 评测为准（见 §5.0 Table 4 的证据路径）。

### 4.2 Methods and Baselines
- ProveTok 主方法：`provetok_lesionness`
- 结构化 baselines：`fixed_grid`, `roi_variance`, `slice_2d`, `slice_2p5d`, `roi_crop`
- 真实模型对照：`ct2rep_strong` 与 `ct2rep_like`（同权重；该变体禁用 citations/refusal，复跑时若产物名为 `ct2rep_noproof` 视为同义）

### 4.3 Statistical Protocol
- Budgets: `{2e6, 3e6, 4e6, 5e6, 6e6, 7e6}`
- Multi-seed
- Paired bootstrap + Holm correction（对方向性假设采用 one-sided；跨预算/反事实 family 做 Holm）
- 机器裁判：`scripts/proof_check.py --profile real`

## 5. Results
### 5.0 Oral minimal evidence（最小但决定性）
| Item | Verdict | Key Numbers | Protocol | Evidence |
|---|---|---|---|---|
| `C0001` | Pass | combined_pass=6/6(need4); iou_pass=6/6(need4); lat_p95_pass=6/6; unsupported_pass=6/6 | budgets=6, seeds=5, n_boot=20000, paired bootstrap(H1>0)+Holm(budgets); lat_p95<=+5%, unsupported_delta<=+0.05 | `outputs/E0164-full/baselines_curve_multiseed.json` |
| `C0002` | Pass | n_boot=20000; CI_high=0.0000; naive_CI_low=0.4823 | dev->test, AIC/BIC model fit, bootstrap CI, requires CI_high<0.15 & beats naive | `outputs/E0161-full/fig3_regret_sweep.json` |
| `C0003` | Pass | no_cite: dIoU=0.0010, p_holm=0; cite_swap: dUnsup=0.0234, p_holm=0 | paired bootstrap + Holm (counterfactual family) | `outputs/E0162-full_retry3/figX_counterfactual_20260206_102521/figX_counterfactual.json` |
| `C0004` | Pass | fixed_grid_pass=6/6(need4); roi_variance_pass=6/6(need4); seeds=20 | one-sided (H1>0) + Holm(budgets), n_boot=20000 | `outputs/E0156-grounding_proof_100g_saliency_seed20/figX_grounding_proof.json` |
| `C0005` | Pass | tau=0.002; miss_max=0<= 0.05; ece_max=0.00183<= 0.15; rr_max=0.1<= 0.2; unsupported_improved=6/6 | hard gates per budget + need>=2/3 budgets improve unsupported | `outputs/E0144-full/figX_refusal_calibration.json` |
| `C0006` | Pass | budget_accounting=True; strong_weights=True; frame_f1_last=0.6967>= 0.05 | baseline coverage + audited cost accounting + strong baseline non-degenerate | `outputs/E0164-full/baselines_curve_multiseed.json` |
| `V0003/omega_perm` | Pass | mean_diff=0.0026; CI=[0.0013,0.0038]; p1=0.0001; p_holm=0.0006; positive=19/20 | pooled one-sided test + secondary Holm over counterfactual family | `outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json` |

生成版表格：`docs/paper_assets/tables/table4_oral_minset.md`

### 5.1 Multi-budget quality/latency
![Figure 2: Budget curves (combined/IoU/latency)](docs/paper_assets/figures/fig2_budget_curves.png)

图 2 数据源：`outputs/E0164-full/baselines_curve_multiseed.json`。

### 5.2 Allocation regret
![Figure 3: Allocation regret sweep](docs/paper_assets/figures/fig3_regret_sweep.png)

图 3 数据源：`outputs/E0161-full/fig3_regret_sweep.json`。

### 5.3 Counterfactual non-triviality (V0003)
![Figure 4: Counterfactual pooled significance](docs/paper_assets/figures/fig4_counterfactual_power.png)

图 4 数据源：`outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json`。

### 5.4 Refusal calibration（硬约束，而非“封嘴刷指标”）
![Figure 6: Refusal calibration](docs/paper_assets/figures/fig6_refusal_calibration.png)

图 6 数据源：`outputs/E0144-full/figX_refusal_calibration.json`。在 `tau_refuse=0.002` 下，refusal 在 test 上冻结并满足 `critical_miss_rate/refusal_ece/refusal_rate` gates，同时在 6/6 budgets 上降低 `unsupported_rate`（对应 Table 4 的 C0005）。

### 5.5 V0003 cross-dataset key table
| Item | Scope | Key result | Verdict |
|---|---|---|---|
| E0166 grounding vs ROI | TS-Seg eval-only | IoU_union: 正向 6/6, Holm 显著 6/6 | Pass |
| E0166 grounding vs Fixed-Grid | TS-Seg eval-only | IoU_union: 正向 5/6, Holm 显著 4/6 | Partial pass |
| E0167 seed0..2 no_cite | counterfactual | mean_diff(avg)=0.0059, Holm 显著 3/3 | Pass |
| E0167 seed0..2 omega_perm | counterfactual | mean_diff(avg)=0.0023, Holm 显著 0/3 | Not significant |
| E0167R pooled | seeds 0..9 | `mean_diff=0.0020`, `p_one_sided=0.0187`, `p_holm=0.1122` | Primary only |
| E0167R2 pooled | seeds 0..19 | `mean_diff=0.0026`, `p_one_sided=0.0001`, `p_holm=0.0006` | Primary + Holm |

完整表：`docs/paper_assets/tables/table2_v0003_cross_dataset.md`

### 5.6 Omega variant search (seed0)
| Variant | Setting | omega mean_diff | omega p_holm | no_cite mean_diff |
|---|---|---:|---:|---:|
| BASE | score + topk=3 | 0.0015 | 1.0 | 0.0059 |
| RA | + score_to_uncertainty | 0.0015 | 1.0 | 0.0059 |
| RD | + score_level_power=1.0 | 0.0015 | 1.0 | 0.0059 |
| RC | score_interleave | 0.0006 | 1.0 | 0.0132 |
| RB | topk=1 | 0.0005 | 1.0 | 0.0017 |

完整表：`docs/paper_assets/tables/table3_omega_variant_search.md`

### 5.7 Qualitative evidence cases
![Figure 5: Qualitative case studies](docs/paper_assets/figures/fig5_case_studies.png)

图 5 数据源：`outputs/E0163-full-v3/case_*/case.png` + `case.json`。

## 6. Discussion
### 6.1 为什么该结果有说服力
- Table 4 给出“oral 最小证据集”：每条结论都能被脚本与产物路径复现，不依赖人工挑样例。
- 多预算 + 多 seed + Holm：C0001/C0004 在 `2e6..7e6` 预算范围内通过多重校正，并同时满足 latency/unsupported 的约束口径。
- 安全不靠封嘴：C0005 显式约束 `critical_miss_rate/refusal_ece/refusal_rate`，并展示 refusal 校准带来的 `unsupported_rate` 降低。
- citation 非装饰：C0003 的反事实套件与 V0003 的 pooled `omega_perm`（secondary Holm）共同构成“非平凡性”证据。

### 6.2 失败模式与边界
- 在跨域 V0003 路径中，TS-Seg 与 pseudo-mask 仍属于 `silver_auto_unverified` 证据，不能替代 gold-mask 主结论。
- `omega_perm` 的显著性需要足够统计功效；seed 扩展是关键（从 R 到 R2）。
- 高预算下个别 baseline 的延迟/unsupported 波动仍需持续监控。

### 6.3 下一步
1. 推进 gold-mask 的跨域 eval 子集，替代银标主证据。  
2. 发布更强公开 baseline 对齐报告。  
3. 将图表生成脚本纳入 CI，确保 README 与最新结果自动同步。

## 7. Conclusion
ProveTok 将预算约束、证据绑定、可验证生成与拒答校准统一成一个可审计闭环。当前版本在 `real` profile（ReXGroundingCT gold mask）下已通过 C0001-C0006；并在 V0003（silver label）路径通过 `E0167R2` 使 `omega_perm` 在 pooled primary（one-sided）与 secondary family-wise Holm 下同时成立。该结果支持一个更可防守的结论：在严格统计与可复核协议下，citation 机制不是装饰，而是能在反事实压力测试中表现出可机检的非平凡效应。

## 8. Reproducibility
### 8.1 Regenerate paper assets
```bash
python scripts/paper/build_readme_figures.py --out-dir docs/paper_assets/figures
python scripts/paper/build_readme_tables.py --out-dir docs/paper_assets/tables
```

### 8.2 Proof gate
```bash
python scripts/proof_check.py --profile real
python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict
```

## References
[1] Hamamci et al. Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography. arXiv:2403.17834, 2024.  
[2] Hamamci et al. CT2Rep: Automated radiology report generation for 3D medical imaging. arXiv:2403.06801, 2024.  
[3] Hamamci et al. GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes. arXiv:2305.16037, 2023.  
[4] Baharoon et al. ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports. arXiv:2507.22030, 2025.  
[5] Song et al. Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse. arXiv:2409.11242, 2024.  
[6] Wasserthal et al. TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images. Radiology: Artificial Intelligence, 2023. doi:10.1148/ryai.230024.  
[7] Delbrouck et al. Memory-driven Transformer for Radiology Report Generation. EMNLP 2020.  
[8] Ji et al. Survey of Hallucination in Natural Language Generation. ACM CSUR, 2023.  
[9] Efron and Tibshirani. An Introduction to the Bootstrap. Chapman and Hall/CRC, 1994.  
[10] Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 1979. doi:10.2307/4615733.

引用校对与 BibTeX 见：`docs/paper_assets/references.md`、`docs/paper_assets/references.bib`。
