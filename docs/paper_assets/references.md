# Paper-Style README References (Verified Notes)

本文件用于记录 `README.md` 里的引用条目（[1]..）的“可检索来源 + 为什么引用”，方便 oral 现场被追问时快速定位依据与边界。

## [1] CT-RATE / CT-CLIP / CT-CHAT（3D CT multimodal foundation）
- Bibliography: Hamamci et al. *Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography*. arXiv:2403.17834 (2024).
- Why here: 提供 CT-RATE 数据与 3D CT multimodal foundation 的背景，解释我们在 V0003 跨域路径中为何选择 CT-RATE 作为弱标/外部域的评测土壤。

## [2] CT2Rep（3D CT report generation baseline）
- Bibliography: Hamamci et al. *CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging*. arXiv:2403.06801 (2024).
- Why here: 说明“3D CT 报告生成”方向的代表性强基线，并强调本文贡献不在于再造一个结构，而在于 proof-carrying 协议。

## [3] GenerateCT（3D CT text-conditional generation）
- Bibliography: Hamamci et al. *GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes*. arXiv:2305.16037 (2023).
- Why here: 说明 3D CT 文本条件生成与“报告生成/证据绑定”不同，避免把生成式指标与 grounding/可信门槛混为一谈。

## [4] ReXGroundingCT（sentence-level grounding in 3D CT）
- Bibliography: Baharoon et al. *ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports*. arXiv:2507.22030 (2025).
- Why here: 这是本文 grounding 口径的核心外部土壤（sentence-level pixel mask），使 IoU/Dice/hit-rate 成为“可验证证据”的一等指标。

## [5] Trust-Align（grounded attribution + learning to refuse）
- Bibliography: Song et al. *Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse*. arXiv:2409.11242 (2024).
- Why here: 用于定位“引用质量 + 拒答”作为可信生成的一部分，并解释我们为何把 refusal calibration 写入硬 gate。

## [6] TotalSegmentator（TS-Seg 外部 pseudo-mask 证据链）
- Bibliography: Wasserthal et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*. Radiology: Artificial Intelligence (2023). doi:10.1148/ryai.230024.
- Why here: V0003 的 TS-Seg 路径属于 `silver_auto_unverified`，必须明确其作用仅为跨域弱证据与 sanity，不可替代 gold-mask 的主结论。

## [7] Memory-driven Transformer（早期 report gen 代表）
- Bibliography: Delbrouck et al. *A Memory-driven Transformer for Radiology Report Generation*. EMNLP 2020.
- Why here: 用作“报告生成”方向的经典基线语境，不作为本文主对照。

## [8] Hallucination survey（NLG 幻觉综述）
- Bibliography: Ji et al. *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys (CSUR), 2023.
- Why here: 用于说明医学场景中 hallucination 风险与“可审计引用 + 拒答”动机。

## [9] Bootstrap（统计协议）
- Bibliography: Efron and Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall/CRC, 1994.
- Why here: paired bootstrap 的理论出处，支撑我们在多预算/多 seeds 下的统计口径。

## [10] Holm correction（多重检验）
- Bibliography: Holm. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 1979. doi:10.2307/4615733.
- Why here: 我们在多 budget、多对照情形使用 Holm 控制 family-wise error，避免“挑预算/挑指标”。
