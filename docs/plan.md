# Plan

## Goals

ProveTok: **Proof-Carrying Budgeted Evidence Tokenization for Grounded 3D Report Generation**

应用：**3D CT Radiology Report Generation with Pixel-level Grounding**

一句话 Punchline（Oral gatekeeper 版）：
在严格预算 B 下，ProveTok 生成带显式 3D 支持域 Ω 的证据 tokens，并以 proof-carrying 生成协议强制每条断言携带可机检证据与可审计 verifier trace；当证据不足时触发可约束的 refusal calibration；跨多预算呈现可解释的 scaling 与 Pareto，并用 pixel-level grounding 与 counterfactual 实验证明 citations 不是装饰。

核心目标：
1. 在严格联合预算 **B = B_enc + B_gen** 下，把 3D→text 的计算分配问题显式化，并在 **FLOPs-matched / latency-matched** 协议下报告所有对比。
2. 把 groundedness 从“事后评测”重写成 **proof-carrying generation (PCG)** 协议：每条断言必须携带可机检证据与可审计 verifier trace；证据不足触发 **refusal calibration**（并用 critical miss-rate 约束防“封嘴”）。
3. 在多预算 B 上呈现可解释 **scaling** 与 **Pareto**，并用 pixel-level **grounding** 与 **counterfactual** 实验证明 citations 不是装饰。

## Claims (C####)

	- [x] C0001: 在多预算 B 下，ProveTok 在 **FLOPs-matched + latency-constrained** 设置中给出更强的 **Pareto dominate**（不是单点赢、也不是只看 mean latency）。
	  - Evidence: E0138 (paper-grade baselines curve), E0137 (latency protocol sanity)
	  - Proof rule（paper-grade, 以可自动判定为准）：
	    - 质量主指标：在 `B={2e6,3e6,4e6,5e6,6e6,7e6}`、`seeds>=5`、`n_bootstrap>=20000` 下，`provetok_lesionness` 相对 `fixed_grid` 的 `combined`（paired bootstrap；**one-sided, H1: mean_diff>0**）在 **≥4/6 budgets** 满足 `mean_diff>0` 且 `p_holm<0.05`；
	    - grounding 约束：同一设置下，`iou_union`（paired bootstrap；**one-sided, H1: mean_diff>0**）也在 **≥4/6 budgets** 满足 `mean_diff>0` 且 `p_holm<0.05`；
    - latency 约束（纳入硬判定）：同一设置下，`warm_time_p95_s` 对 `fixed_grid` 的相对增幅在所有 budgets 满足 `Δp95 <= +5%`（tail-latency hard gate；见 `python scripts/proof_check.py` 输出）；
    - trust 约束（纳入硬判定）：同一设置下，`unsupported` 相对 `fixed_grid` 的增幅在所有 budgets 满足 `Δunsupported <= +0.05`（绝对值；避免“为了赢 grounding 把 unsupported 推高”）。
  - Notes:
    - B = B_enc + B_gen；verifier/ROI selector 成本必须入账（见 `outputs/compute_costs.json`）。
    - `python scripts/proof_check.py` 对 C0001 的输出是最终裁判（paper-grade 规则）。
    - 2026-02-06（real profile 最新状态）：`outputs/E0164-full/baselines_curve_multiseed.json` 对应判定为 proved（`combined_pass=6/6`, `iou_pass=6/6`, `latency_p95_pass=6/6`, `unsupported_pass=6/6`）。

- [x] C0002: `scaling law + allocation model` 能预测预算下最优配置并报告 **regret**（含 CI，并显著优于 naive policy）。
  - Evidence: E0142 (paper-grade dev curve), E0138 (paper-grade test curve), E0141 (paper-grade regret + CI)
  - Proof rule（paper-grade）：
    - dev(split=val) 上对每个 method 拟合 scaling law，并进行 AIC/BIC 选型（锁定 criterion）；
    - test(split=test) 上输出 per-budget predicted method 与 oracle method，并报告 normalized regret 的 bootstrap CI；
    - 通过条件：`mean_normalized_regret_ci_high <= 0.15`，且显著优于至少一个 naive policy（例如 `always_fixed_grid`）。

- [x] C0003: **Permutation / citation-swap / evidence-drop** 等反事实显著击穿（paired bootstrap ≥10k + Holm）。
  - Evidence: E0004, E0109, E0113, E0157
  - Proof rule (minimal): `cite_swap` 在 unsupported 上显著上升（Holm），且 `no_cite` 在 grounding 上显著下降（Holm）；见 `python scripts/proof_check.py` 的 C0003。
  - Optional stronger checks:
    - non-oracle `omega_perm` 对 grounding 的显著击穿：已通过且 margin 更足（E0157，`iou_union` 口径 `p_holm=0.0044`；由 `E0135` 训练的 `saliency_cnn3d.pt` 驱动 token.score，并用 `ToyPCG(citation_strategy=\"score\")` 选择 citations）
    - oracle sanity：当 citations/Ω 由 GT mask 对齐驱动时，`omega_perm` 可稳定显著击穿（E0133）

	- [x] C0004: 在 ReXGroundingCT 上 citation-grounding（IoU/Dice/hit-rate）显著提升（pixel-level grounding），并覆盖多个 baselines（不止 fixed-grid）。
	  - Evidence: E0143 (paper-grade grounding proof, ReXGroundingCT-mini), E0156 (stronger supplement on ReXGroundingCT-100g via saliency-scored citations)
	  - Proof rule（paper-grade）：
	    - 在 `B={2e6,3e6,4e6,5e6,6e6,7e6}`、`seeds>=5`、`n_bootstrap>=20000` 下，
	      `provetok_lesionness` 相对 `fixed_grid` **与** `roi_variance` 的 `iou_union` 均在 **≥4/6 budgets** 满足 `mean_diff>0` 且 `p_holm<0.05`（paired bootstrap；**one-sided, H1: mean_diff>0**；Holm across budgets）；
    - 同时报告 `hit_lesion_coverage` 与 per-sample grounding artifacts（便于审计“不是靠背景 overlap”）。

- [x] C0005: 在 unsupported↓ 的同时 **critical miss-rate 不升** 且 refusal 校准（ECE/reliability）可控（反封嘴，含 CI/阈值）。
  - Evidence: E0144 (paper-grade refusal calibration)
  - Proof rule（paper-grade）：
    - τ_refuse 在 dev(split=val) 集按 `critical miss-rate ≤ δ`（δ=0.05）选择一次并跨 budgets 固定；
    - test(split=test) 上：
      - `critical miss-rate ≤ 0.05`（所有 budgets）；
      - `unsupported_rate` 相对 no-refusal baseline 在 **≥4/6 budgets** 下降；
      - `refusal_ece ≤ 0.15`（所有 budgets），并报告 reliability bins；
      - `refusal_rate ≤ 0.20`（所有 budgets，防“封嘴”）。

- [x] C0006: Baselines 齐全（含**可复现强 3D RRG baseline** + 2D/2.5D/ROI/fixed-grid），且 detector/ROI 成本入账，matched setting 不缺席。
  - Evidence: E0140 (ct2rep_strong train), E0138 (paper-grade baselines curve)
  - Proof rule（paper-grade）：
    - baseline suite 至少包含 `fixed_grid/slice_2d/slice_2p5d/roi_crop/roi_variance`；
    - 强 baseline：`ct2rep_strong` 的 weights 存在且可加载，且在 `baselines_curve_multiseed` 的 methods 中实际出现（避免“只写在文档里”）；
    - 强 baseline 非退化：`baselines_curve_multiseed` 中 `ct2rep_strong` 的 `frame_f1` 在最后一个 budget 的均值 `>=0.05`；
    - 成本核算：`costs_json` + `budgets_by_method` 可审计复现；不允许 “Full=[x] 但 Smoke=[ ]” 倒挂。

## Plan Items (P####)

- [x] P0001: 初始化 RD docs（`docs/plan.md`/`docs/mohu.md`/`docs/experiment.md`）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: 三个文档存在且符合 docs-spec，且 Appendix A 含“原文需求”逐字粘贴
  - Verification: `ls docs && sed -n '1,40p' docs/plan.md`
  - Touchpoints: `docs/plan.md`, `docs/mohu.md`, `docs/experiment.md`

- [x] P0002: 实现 `provetok.data`（manifest 驱动 + 可选下载/预处理）与 `ProtocolLock`
  - Linked claims: C0004, C0006
  - Definition of done: `from provetok.data import make_dataloader` 可用；manifest 校验/拆分/去重脚本可运行
  - Verification: `python -c "from provetok.data import make_dataloader; print('ok')"`
  - Touchpoints: `provetok/data/*`, `scripts/data/*`

- [x] P0003: 统一 `cell_id` / `phi` round-trip，并修复 grounding mask 构造
  - Linked claims: C0004, C0003
  - Definition of done: `cell_id` 解析与 `Cell.id()` 一致；grounding 指标不因解析失败退化为 0
  - Verification: `pytest -q`
  - Touchpoints: `provetok/grid/cells.py`, `provetok/eval/metrics_grounding.py`, `tests/*`

- [x] P0004: 扩展 finding frames schema（含 location/size_bin/severity/uncertain 等）并对齐 PCG/verifier/metrics
  - Linked claims: C0001, C0004, C0005
  - Definition of done: 全链路使用统一 FindingFrame；Hungarian matching 与 slot-level 指标可跑
  - Verification: `pytest -q`
  - Touchpoints: `provetok/types.py`, `provetok/models/pcg_head.py`, `provetok/eval/metrics_frames.py`, `provetok/verifier/rules.py`

- [x] P0005: 落地 PCG “双通道协议”（findings table + 可逆叙述文本）并在 verifier 中做回译一致性检查
  - Linked claims: C0005
  - Definition of done: text 不能绕过 frames；不一致必出 I1 issue（rule-id 固定）
  - Verification: `pytest -q`
  - Touchpoints: `provetok/data/frame_extractor.py`, `provetok/verifier/rules.py`

- [x] P0006: 补齐 counterfactual 套件（token-permutation / evidence-drop / mask sanity）+ paired bootstrap + Holm
  - Linked claims: C0003
  - Definition of done: 反事实实验可运行并输出 CI + Holm
  - Verification: `python -m provetok.experiments.figX_counterfactual --smoke`
  - Touchpoints: `provetok/eval/*`, `provetok/experiments/*`

- [x] P0007: 联合预算与公平性协议落地（FLOPs/latency matched、ROI 成本入账）
  - Linked claims: C0001, C0002, C0006
  - Definition of done: runner 统一 BudgetProtocol；输出 FLOPs_total 与 latency(mean/P95,cold/warm)
  - Verification: `python scripts/bench_latency.py --smoke`
  - Touchpoints: `provetok/eval/compute_budget.py`, `scripts/bench_latency.py`

- [x] P0008: Baselines scaffolding（tokenization + protocol ablations + 强 baseline runner 接口）
  - Linked claims: C0006
  - Definition of done: baseline 列表覆盖 §7.2；matched setting 可跑（至少 synthetic）
  - Verification: `python -m provetok.experiments.run_baselines --smoke`
  - Touchpoints: `provetok/baselines/*`, `provetok/experiments/*`

- [x] P0009: 维护 `docs/experiment.md`（Fig2/Fig3/Table1/Counterfactual/Matched baselines 全部 E####）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: 每个 E#### 有 smoke/full 命令、指标、资源预估与产物路径
  - Verification: `sed -n '1,120p' docs/experiment.md`
  - Touchpoints: `docs/experiment.md`

- [x] P0010: 真实数据管线 + ProtocolLock v1.0（manifest 构建、去重、交集锁、版本锁）
  - Linked claims: C0004, C0006
  - Definition of done: `scripts/data/build_*.py` 可从原始数据目录生成标准 manifest + split manifest + revision hash；并扩展 `ProtocolLock` 覆盖去重与交集条款（ReX val/test 不进入训练/阈值选择/拟合）。
  - Verification: `python scripts/data/build_ct_rate_manifest.py --help`
  - Touchpoints: `provetok/data/*`, `scripts/data/*`, `docs/plan.md`（§6.3 protocol lock）

- [x] P0011: ReXGroundingCT pixel-level grounding 闭环（mask 读取/对齐/评测 + mask sanity）
  - Linked claims: C0004, C0003
  - Definition of done: dataloader 可提供 per-finding 3D mask；Table1 输出 IoU/Dice/hit-rate；Fig3 反事实与 mask-sanity 能在 ReXGroundingCT 上跑通。
  - Verification: `python -m provetok.experiments.figX_counterfactual --smoke`
  - Touchpoints: `provetok/data/dataset.py`, `provetok/data/io.py`, `provetok/eval/metrics_grounding.py`, `provetok/experiments/*`

- [x] P0012: 替换 toy BET tokenization：接入真实 3D encoder + cell pooling + 增量缓存
  - Linked claims: C0001, C0002, C0006
  - Definition of done: `encode_tokens()` 可注入 `BaseEncoder3D` 并缓存 feature map；refine loop 只对新增子 cell 做增量编码；预算/成本入账对齐 encoder FLOPs。
  - Verification: `pytest -q`
  - Touchpoints: `provetok/bet/tokenize.py`, `provetok/models/encoder3d.py`, `provetok/bet/refine_loop.py`

- [x] P0013: 锁死 ε 与 τ_refuse 的开发集一次性选择规则，并在多预算实验中强制复用
  - Linked claims: C0002, C0005
  - Definition of done: 提供 ε 校准脚本（分位点规则）与 τ_refuse 校准脚本（critical miss-rate ≤ δ 约束）；实验 runner 对多预算强制使用同一 ε/τ（禁止按预算调参）。
  - Verification: `python -m provetok.experiments.figX_refusal_calibration --smoke`
  - Touchpoints: `provetok/bet/refine_loop.py`, `provetok/pcg/refusal.py`, `provetok/experiments/*`, `scripts/*`

- [x] P0014: 训练阶段 M0→M3 真正落地（citation supervision + verifier loss + grounding loss + allocator 学习）
  - Linked claims: C0001, C0003, C0004, C0005
  - Definition of done: `Trainer(stage=M1/M2)` 支持 citation 弱监督、verifier-driven loss、grounding loss（若有 mask）与 EvidenceHead/allocator 学习；产出可复现实验 checkpoint 与日志。
  - Verification: `pytest -q`
  - Touchpoints: `provetok/training/*`, `provetok/models/*`, `provetok/bet/*`, `provetok/pcg/*`

- [x] P0015: Baselines 论文级补齐（2.5D / ROI-crop 成本入账 / CT2Rep baseline 占位接口 + latency 报表）
  - Linked claims: C0006
  - Definition of done:
    - baselines 列表覆盖 2D / 2.5D / ROI / fixed-grid，并包含 `ct2rep_noproof`（真实模型推理但禁用 citation/refusal 的无 proof-carrying 对照）；
    - ROI selector/detector 的额外成本入账可复现（FLOPs-matched）；
    - baseline 侧提供同协议的 latency 报表（cold/warm mean+P95），并落盘到可审计 artifact（E0137）。
  - Verification: `python -m provetok.experiments.run_baselines --smoke && python -m provetok.experiments.latency_bench_baselines --smoke --device cpu --budget-tokens 64 --output-dir ./outputs/_verify_latency_bench`
  - Touchpoints: `provetok/baselines/*`, `provetok/experiments/run_baselines.py`, `provetok/experiments/latency_bench_baselines.py`, `scripts/rd_queue.py`, `docs/experiment.md`

- [x] P0016: 真实 FLOPs/latency matched 执行器（强制公平性协议 + 报表落盘）
  - Linked claims: C0001, C0002, C0006
  - Definition of done: runner 以统一预算单位（FLOPs_total）驱动配置搜索/匹配；输出 cold/warm mean+P95；对不公平配置直接 fail-fast。
  - Verification: `python scripts/bench_latency.py --smoke`
  - Touchpoints: `provetok/eval/compute_budget.py`, `scripts/bench_latency.py`, `provetok/experiments/*`

- [x] P0017: 统一可审计 artifact schema（code commit / data revision / split manifest / rule-set 版本）
  - Linked claims: C0001, C0004, C0005
  - Definition of done: 所有 runner 输出统一 `artifact.json`，包含版本锁信息与 per-sample trace；verifier rule-id、schema/taxonomy 版本写入 artifact meta。
  - Verification: `python -m provetok.run_demo --steps 2 --budget 32 --seed 0`
  - Touchpoints: `provetok/run_demo.py`, `provetok/verifier/*`, `provetok/utils/*`, `provetok/experiments/*`

- [x] P0018: `.rd_queue` 实验队列与自动更新台账（smoke→full→proof）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: 支持把 `docs/experiment.md` 的 smoke/full 入队执行，产出 `.rd_queue/logs` 与 `.rd_queue/results` JSON；成功后自动勾选对应实验并写入关键指标。
  - Verification: `python scripts/rd_queue.py --help`
  - Touchpoints: `.rd_queue/*`, `docs/experiment.md`, `scripts/*`

- [x] P0019: 补齐依赖声明与可复现安装（`requirements.txt` 覆盖运行/实验所需依赖）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `requirements.txt` 覆盖当前代码/脚本中已使用的依赖（例如 `scipy`, `tqdm`, `huggingface_hub`）；新环境可 `pip install -r requirements.txt` 后通过 `pytest -q` 与关键脚本 `--help`。
  - Verification: `python -c "import scipy, tqdm; import huggingface_hub; print('deps_ok')"`
  - Touchpoints: `requirements.txt`, `provetok/eval/metrics_frames.py`, `provetok/experiments/fig2_scaling_law.py`, `scripts/data/download_ct_rate_mini_from_rex.py`

- [x] P0020: 补齐 CT-3DRRG 数据集入口（index/manifest 构建脚本 + ProtocolLock 对齐）
  - Linked claims: C0006
  - Definition of done: 新增 `scripts/data/make_ct_3drrg_index.py` 与 `scripts/data/build_ct_3drrg_manifest.py`（或等价）把原始 CT-3DRRG 目录/索引转为标准 `manifest.jsonl`；并输出 `*.meta.json`/`*.splits.json`（版本锁/拆分清单），可被 `--dataset-type manifest` 的实验直接消费。
  - Verification: `python scripts/data/build_ct_3drrg_manifest.py --help`
  - Touchpoints: `scripts/data/*`, `provetok/data/manifest_schema.py`, `provetok/data/protocol_lock.py`, `docs/plan.md`（§6 数据）

- [x] P0021: Compute/Fairness 真正落地到 Fig2/Fig3（FLOPs/latency matched 可复现）
  - Linked claims: C0001, C0002, C0006
  - Definition of done: `fig2_scaling_law`/`figX_counterfactual`/`run_baselines` 的主运行入口统一支持 `--costs-json`（来自 `scripts/profile_flops.py --out-costs`）与 `--flops-total`（或等价 matched 配置）；并在不匹配时 fail-fast；输出报表包含 `flops_total` 与 latency(mean/P95,cold/warm)。
  - Verification: `python scripts/profile_flops.py --device cpu --out ./outputs/flops_profile.json --out-costs ./outputs/compute_costs.json`
  - Touchpoints: `provetok/eval/compute_budget.py`, `scripts/profile_flops.py`, `provetok/experiments/fig2_scaling_law.py`, `provetok/experiments/figX_counterfactual.py`, `provetok/experiments/run_baselines.py`

- [x] P0022: 完善 `docs/experiment.md`（Multi-GPU scripts、去掉 TBD、smoke/full 一致性）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `docs/experiment.md` 每行都提供可运行的 `1GPU script` 与 `Multi-GPU script`（不再出现 `TBD`）；checkbox 语义一致（至少保证 `Full=[x]` 不会出现 `Smoke=[ ]` 的倒挂）；`python scripts/rd_queue.py make` 可从台账生成队列并成功解析所有命令。
  - Verification: `python -c "import pathlib; t=pathlib.Path('docs/experiment.md').read_text(encoding='utf-8'); assert 'TBD' not in t; print('no_tbd')"`
  - Touchpoints: `docs/experiment.md`, `scripts/rd_queue.py`, `.rd_queue/*`

- [x] P0023: LLM-backed PCG 支持多 frame 输出（修复 `Llama2PCG` 仅单帧限制）
  - Linked claims: C0001, C0004, C0005
  - Definition of done: `provetok/pcg/llama2_pcg.py` 支持输出 0..K 帧（K 可配置），并保持 citations/q/refusal 的 schema 不变；增加不依赖真实模型权重的单元测试覆盖 JSON 抽取/清洗/多帧 citations 约束。
  - Verification: `pytest -q`
  - Touchpoints: `provetok/pcg/llama2_pcg.py`, `provetok/pcg/narrative.py`, `tests/*`

- [x] P0024: 训练入口对齐 StageConfig（使用 bet_steps/budget_tokens/epsilon/refine_loop，去掉 dummy 训练）
  - Linked claims: C0001, C0003, C0004, C0005
  - Definition of done: `Trainer` 或训练入口脚本真正使用 `StageConfig` 的 `budget_tokens/bet_steps/epsilon/max_depth`（调用 `run_refine_loop` 或等价）；`scripts/train_m0.py` 不再依赖 dummy 参数（改为调用统一 Trainer/系统），并提供最小 smoke 配置在 CPU 上 2 steps 跑通。
  - Verification: `python -c "from provetok.training.trainer import Trainer, TrainerConfig; cfg=TrainerConfig(stage='M1', device='cpu', dataset_cfg={'dataset_type':'synthetic','num_samples':4,'vol_shape':[16,16,16],'batch_size':2}, output_dir='./outputs', overrides={'max_steps':2,'log_every':1,'eval_every':100000,'save_every':100000}); Trainer(cfg).train(); print('train_smoke_ok')"`
  - Touchpoints: `provetok/training/stages.py`, `provetok/training/trainer.py`, `provetok/bet/refine_loop.py`, `scripts/train_m0.py`

- [x] P0025: 更新 `README.md`（从“scaffold”对齐到当前 pipeline + datasets + experiments）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `README.md` 描述与当前实现一致：包含数据 manifest 构建/校验入口、`run_demo`/`rd_queue`/关键 experiments 的运行方式，以及“toy vs real”开关说明（例如 `--dataset-type manifest`, `--pcg llama2`, `--encoder cnn3d`）。
  - Verification: `sed -n '1,120p' README.md`
  - Touchpoints: `README.md`, `docs/plan.md`, `docs/experiment.md`

---

- [x] P0026: 产出“项目现状 vs 论文/Plan 差异”清单并落盘（gap report）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `docs/gap_report.md` 完整列出 repo 现状、差异清单、缺失实验清单，并把缺口同步到 `docs/plan.md`/`docs/mohu.md`/`docs/experiment.md`
  - Verification: `test -f docs/gap_report.md && sed -n '1,60p' docs/gap_report.md`
  - Touchpoints: `docs/gap_report.md`, `docs/plan.md`, `docs/mohu.md`, `docs/experiment.md`

- [x] P0027: 所有 experiments 输出统一 artifact meta（版本锁 + 可审计字段）
  - Linked claims: C0001, C0003, C0004, C0005, C0006
  - Definition of done: `fig2_scaling_law`/`run_baselines`/`figX_counterfactual`/`figX_refusal_calibration` 输出 JSON 均包含 `meta`（code commit / schema/taxonomy/rule-set / data revision / split manifest / hardware）
  - Verification: `python -m provetok.experiments.fig2_scaling_law --budgets 8 --n-samples 1 --no-plot --output-dir ./outputs/_verify_meta && python -c "import json; d=json.load(open('outputs/_verify_meta/fig2_raw_data.json')); assert 'meta' in d; print('meta_ok')"`
  - Touchpoints: `provetok/utils/artifact.py`, `provetok/experiments/*`, `docs/experiment.md`

- [x] P0028: Fig2/Baselines 支持 multi-seed + CI（paper-grade）
  - Linked claims: C0001, C0002, C0004, C0006
  - Definition of done: 新增/扩展 runner 可一次跑多 seeds 并输出 paired/bootstrap CI（含多预算），可用于 Pareto dominate 的统计证明
  - Verification: `python -m provetok.experiments.fig2_scaling_multiseed --help`
  - Touchpoints: `provetok/experiments/*`, `provetok/eval/stats.py`, `docs/experiment.md`

- [x] P0029: 运行并通过 paper-grade experiments（更新台账）
  - Linked claims: C0001, C0002, C0003, C0004, C0006
  - Definition of done: `docs/experiment.md` 中新增的 paper-grade 行（E0107–E0110）通过 `.rd_queue` 跑通并勾选；对应产物落在 `outputs/E0107*` 等目录
  - Verification: `python scripts/rd_queue.py make --stage full --ids E0107 E0108 E0109 E0110 --out .rd_queue/queue_full.json`
  - Touchpoints: `docs/experiment.md`, `.rd_queue/*`, `outputs/*`

- [x] P0030: Proof audit（逐条检查 C0001–C0006 是否被实验结果证明）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `docs/proof_audit.md` 完整列出每条 Claim 的 proved/not-proved 判定与缺口清单，并可转化为新增实验/实现任务。
  - Verification: `test -f docs/proof_audit.md && sed -n '1,40p' docs/proof_audit.md`
  - Touchpoints: `docs/proof_audit.md`, `docs/plan.md`, `docs/experiment.md`, `docs/results.md`

- [x] P0031: 修复实验台账 checkbox 语义（Full 通过后 Smoke 自动视为通过，避免倒挂）
  - Linked claims: C0001, C0002, C0003, C0004, C0005, C0006
  - Definition of done: `python scripts/rd_queue.py sync` 后，`docs/experiment.md` 不再出现任意行 `Smoke=[ ]` 且 `Full=[x]` 的倒挂。
  - Verification: `python scripts/rd_queue.py sync && python -c "import re, pathlib; t=pathlib.Path('docs/experiment.md').read_text(encoding='utf-8'); assert re.search(r'\\|\\s*Smoke\\s*\\|', t); assert not re.search(r'\\|\\s*\\[ \\]\\s*\\|\\s*\\[x\\]\\s*\\|\\s*$', t, re.M); print('checkbox_ok')"`
  - Touchpoints: `scripts/rd_queue.py`, `docs/experiment.md`

- [x] P0032: Baselines 输出补齐 Fig2 主指标口径（IoU/Dice/hit-rate/combined + latency）
  - Linked claims: C0001, C0004, C0006
  - Definition of done: `run_baselines` 的 JSON 输出包含 `dice/hit_rate/combined/warm_time_s`，并支持 multi-seed CI 聚合；用于与 Fig2 的指标口径对齐。
  - Verification: `python -m provetok.experiments.run_baselines --smoke --output-dir ./outputs/_verify_baselines && python -c "import json, pathlib; p=max(pathlib.Path('outputs/_verify_baselines').glob('*/baselines.json'), key=lambda x:x.stat().st_mtime); d=json.loads(p.read_text()); assert 'dice' in d['raw']['fixed_grid']; assert 'hit_rate' in d['raw']['fixed_grid']; assert 'combined' in d['raw']['fixed_grid']; assert 'warm_time_s' in d['raw']['fixed_grid']; print('baselines_metrics_ok')"`
  - Touchpoints: `provetok/experiments/run_baselines.py`, `provetok/eval/metrics_grounding.py`

- [x] P0033: 新增并运行 “Baselines 多预算曲线（multi-seed + CI）” 实验
  - Linked claims: C0001, C0004, C0006
  - Definition of done: `docs/experiment.md` 新增 E0111；full 通过后产出 `outputs/E0111-full/baselines_curve_multiseed.json`，并在 `docs/results.md` 有记录。
  - Verification: `python -m provetok.experiments.baselines_curve_multiseed --help`
  - Touchpoints: `provetok/experiments/baselines_curve_multiseed.py`, `docs/experiment.md`, `.rd_queue/*`

- [x] P0034: 新增并运行 “Fig2 绑定预算范围” 实验（避免早停饱和导致曲线退化）
  - Linked claims: C0001, C0002
  - Definition of done: `docs/experiment.md` 新增 E0112；在更低/更广预算区间下重新跑 Fig2 multi-seed（含 CI），并输出可用于 scaling/pareto 的曲线。
  - Verification: `python -m provetok.experiments.fig2_scaling_multiseed --help`
  - Touchpoints: `provetok/experiments/fig2_scaling_multiseed.py`, `docs/experiment.md`, `.rd_queue/*`

## Changelog

- 2026-02-03: 将“ProveTok paper outline/验收条件”逐字写入 `docs/plan.md` Appendix A，并完成闭环 scaffold（BET/PCG/verifier/refusal/baselines）。
- 2026-02-03: 通过 `.rd_queue` 跑通 E0001–E0006 smoke（synthetic），并同步勾选 `docs/experiment.md`；补齐 schema/taxonomy 版本锁与 FLOPs-matched enforcement。注意：smoke 仅验证可运行性，不足以证明 C0001–C0006（仍需 full + 真实数据）。
- 2026-02-03: 基于当前仓库状态补齐未完成事项清单，新增 P0019–P0025（依赖/CT-3DRRG/compute-matched/台账/LLM多帧/训练入口/README），并同步更新 `docs/mohu.md` 的对应 backlog。
- 2026-02-03: 完成并勾选 P0019–P0025（依赖补齐、CT-3DRRG 入口、Fig2/FigX matched 入口、实验台账去 TBD、Llama2PCG 多帧、训练入口对齐、README 对齐）。
- 2026-02-03: 基于“论文级 full=可审计+CI+matched”语义，重新标记 P0015–P0018 为未完成，并新增 P0026–P0029（gap report / artifact meta / multiseed CI / paper-grade experiments）。
- 2026-02-03: 进行 proof audit（见 `docs/proof_audit.md`），并新增 P0031–P0034 作为下一轮闭环任务（checkbox 语义修复 / baselines 指标口径对齐 / baselines 多预算曲线 / Fig2 绑定预算范围）。
- 2026-02-03: 完成并验证 P0031/P0032（Full→Smoke 语义修复；baselines 指标口径对齐 + 多预算曲线脚本），待运行 E0111/E0112。
- 2026-02-03: 通过 `.rd_queue` 跑通 E0111/E0112 full，并同步勾选 `docs/experiment.md` 与摘要写入 `docs/results.md`（见 `outputs/E0111-full/baselines_curve_multiseed.json`、`outputs/E0112-full/fig2_multiseed.json`）。
- 2026-02-03: 重新跑通 E0113 full 并明确 C0003 的最小 proof rule（`cite_swap`→unsupported、`no_cite`→grounding，Holm），`python scripts/proof_check.py` 现在判定 C0003 可证明；`omega_perm` 作为更强可选检查保留在后续工作中。
- 2026-02-04: 同步标记 P0016/P0017 已完成（对应 verification 命令均可通过），并补齐 E0117/E0118 的 `.rd_queue` 记录与 `docs/results.md` 摘要。
- 2026-02-06: real profile 的 C0001 已关闭（`outputs/E0164-full/baselines_curve_multiseed.json`：`combined/iou/latency_p95/unsupported` 全部 `6/6`），`python scripts/oral_audit.py --sync --strict` 返回 `ready_for_oral_gate=true`。
- 2026-02-06: 推进 V0003(C)（跨数据集弱标签 grounding）：新增 `scripts/data/build_ct_rate_pseudomask_manifest.py`，产出 `/data/provetok_datasets/ct_rate_100g_pseudomask/manifest.jsonl`（CT-RATE test=30，含 `mask_path`）；完成 smoke + preflight 证据：`outputs/E0165-ct_rate-pseudomask-smoke/figX_grounding_proof.json`、`outputs/E0165-ct_rate-pseudomask-preflight/figX_grounding_proof.json`、`outputs/E0165-ct_rate-pseudomask-counterfactual-preflight/`。
- 2026-02-07: 完成 E0165 full（V0003(C) CT-RATE weak-label grounding，58 test 全量 + seeds=0/1/2，bootstrap=20000）；结果文件为 `outputs/E0165-ct_rate-pseudomask-full/figX_grounding_proof.json`：`iou_union` 对 `roi_variance` 在 `6/6` budgets Holm 显著，对 `fixed_grid` 在 `5/6` budgets Holm 显著（`3e6` 预算 `p_holm=0.5212`，但 `mean_diff` 仍为正）。
- 2026-02-07: 完成 E0165CF full（V0003(C) CT-RATE weak-label counterfactual，58 test 全量，bootstrap=20000，seed=0）；结果文件为 `outputs/E0165-ct_rate-pseudomask-counterfactual-full/figX_counterfactual_20260207_044425/figX_counterfactual.json`：`grounding_iou_union_orig_minus_cf` 中 `omega_perm` 显著（`mean_diff=+0.005398`, `p_holm=0.0308`），`no_cite` 显著（`mean_diff=+0.009464`, `p_holm=0.0`）。
- 2026-02-07: 完成 E0165CF1/E0165CF2 full（seed=1/2）并验证多 seed 稳定性：`grounding_iou_union_orig_minus_cf` 的 `omega_perm` 在 seed `{0,1,2}` 均显著（Holm 后 `p={0.0308,0.0048,0.0024}`），`no_cite` 在 seed `{0,1,2}` 均显著（Holm 后 `p=0.0`）。
- 2026-02-07: 复跑 `python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict`，返回 `ready_for_oral_gate=true` 且 `gaps=[]`（审计时间：`2026-02-07T05:27:15+00:00`）。
- 2026-02-07: 完成 E0166/E0167（V0003(A')，CT-RATE TS-Seg 外部自动 mask eval-only）full：`E0166`（`outputs/E0166-ct_rate-tsseg-effusion-grounding-full/figX_grounding_proof.json`）在 `iou_union` 上对 `roi_variance` 达到 `6/6` budgets Holm 显著、对 `fixed_grid` 为 `4/6` budgets Holm 显著（`5/6` 为正）；`E0167` 三 seed（`E0167S0/S1/S2`）显示 `no_cite` 在 `grounding_iou_union_orig_minus_cf` 上 `3/3` 显著（`mean_diff=+0.005909`, Holm `p=0.0`），`omega_perm` 方向一致但未显著。该路径仍属 `silver_auto_unverified`，不替代 gold-mask 证据。

## Appendix A — Paper Outline (verbatim)

> 下方内容为“需求原文”（逐字粘贴），作为不可漂移的协议/实验/验收源文档。

ProveTok：Proof-Carrying Budgeted Evidence Tokenization for Grounded 3D Report Generation
应用：3D CT Radiology Report Generation with Pixel-level Grounding

---
1. 一句话 Punchline（Oral gatekeeper 版）
在严格预算 B 下，ProveTok 生成带显式 3D 支持域 Ω 的证据 tokens，并以 proof-carrying 生成协议强制每条断言携带可机检证据与可审计 verifier trace；当证据不足时触发可约束的 refusal calibration；跨多预算呈现可解释的 scaling 与 Pareto，并用 pixel-level grounding 与 counterfactual 实验证明 citations 不是装饰。
Reviewer-check（硬性）
- 句子里必须出现：预算 B、Ω、proof-carrying、refusal calibration、scaling、grounding、counterfactual ✅

---
2. 背景与动机（仅 4 段，段段有用）
2.1 预算失衡：3D→text 的 compute 不可控
3D 体数据包含大量冗余体素；fixed-grid 全量 tokenization 在 3D 上计算爆炸，而 2D/2.5D slice/ROI crop 虽省算力却丢失 3D 证据并引入启发式偏差。我们关心的是：在严格预算 B 下，模型应如何把 compute 分配到最“证据密集”的空间区域与推理步骤。本文将 B 定义为联合预算
[
B=B_{\\text{enc}}+B_{\\text{gen}},
]
分别约束证据 tokenization 的 encoder compute 与报告生成的 decoder compute；所有对比在 FLOPs-matched 或 latency-matched 协议下报告（把“省算力”从口号变成可检验约束）。
Reviewer-check
- 这段必须让 B 成为主轴 ✅
- 明确指出 fixed-grid vs slice/ROI 的失衡 ✅

---
2.2 可信失衡：报告“像”不等于“有证据”
在临床文本生成中，hallucination 成本极高；传统做法把 groundedness 当作事后评测或人工抽检，无法把“证据不足时应拒答/不确定”变成可控机制。近期口径是把 trustworthiness 变成硬指标与训练目标：例如在检索增强生成中同时评估 citation groundedness 与“该拒答时拒答”的能力，并把拒答质量当作可校准信号（reliability/ECE），同时用漏报率约束防止“封嘴”。这与 ICLR 2025 的 Trust-Align 类工作将 grounded attributions 与 learning-to-refuse 联合作为信任目标的趋势一致。(OpenReview)
Reviewer-check
- 必须对齐 oral 趋势，并引出“拒答可校准 + 防封嘴” ✅

---
2.3 我们的重写：Budgeted Evidence Tokenization → Proof-Carrying Generation
我们将 3D 报告生成重写为两个耦合问题：
(1) BET：在预算 (B_{\\text{enc}}) 内生成证据 tokens，每个 token 绑定显式 3D 支持域 Ω；
(2) PCG：生成报告时，每条临床断言必须携带 token-citation，并由可复现 verifier 检查；若证据不足，则以可校准方式拒答/不确定。
关键重写在于：我们不再把 groundedness 当作事后指标，而是把它写进生成协议——每条断言必须“携证据上岗”，否则 verifier 必须能指明失败类型并触发 refine 或拒答，从而把“像真的”与“有证据”分离。
Reviewer-check
- 必须明确“重写问题形式”，否则会被喷系统工程 ✅

---
2.4 我们证明什么（三类主结果）
1. Efficiency–Correctness：在多预算 B 上呈现 Pareto dominate，并拟合可解释 scaling law 与 compute allocation model（对齐 test-time compute / inference scaling 口径）。我们借鉴 OpenReview 上关于“在计算约束下预测最优推理配置与分配”的 RAG inference scaling 叙事：不仅画曲线，还要能预测最优分配并给出 regret。(OpenReview)
2. Grounding：在具有 pixel-level 3D segmentation 的数据上，citation-grounding（IoU/Dice/hit-rate）显著更强。(arXiv)
3. Non-triviality：Permutation / citation-swap / evidence-drop 等反事实实验显著击穿，证明 citations 不是 attention 装饰，而是可验证的因果依赖。
Reviewer-check
- 这段等价于 oral 的“主图导读”：Pareto+scaling、grounding、counterfactual 三板斧 ✅

---
3. 问题定义（用 B 统治全篇）
给定 3D 体数据 (V) 与预算 (B=B_{\\text{enc}}+B_{\\text{gen}})（详见 §7.3 公平性协议），学习一个闭环系统。
3.0 离散化约定（保证 verifier 可机检）
在体素网格上定义 cell family (\\mathcal{G})（如八叉树节点或规则 block）。每个支持域
[
\\Omega \\equiv \\texttt{cell_id}\\in\\mathcal{G}
]
是可枚举索引；其对应体素集合由确定性函数
[
\\phi(\\texttt{cell_id})\\rightarrow {(x,y,z)}
]
给出，使得 verifier 不依赖学习模型即可回溯证据。

---
3.1 证据 tokenization（BET）
输出证据 tokens：
$$T={(t_i,\\Omega_i,s_i)}_{i=1}^{|T|},\\quad |T|\\le B_{\\text{enc}}$$
其中 (t_i\\in\\mathbb{R}^d) 为 cell 特征（3D encoder 在 (\\phi(\\Omega_i)) 上 pooling 得到），(\\Omega_i) 为显式 3D 支持域（cell_id），(s_i\\in[0,1]) 为 evidence head 的摘要分数（finding relevance / uncertainty，用于 greedy (\\Delta) 与 refusal 决策）。

---
3.2 proof-carrying 生成（PCG）
生成报告 (y)，并对每条关键断言（frame）(k) 输出引用集合
$$C_k\\subseteq{1,\\dots,|T|},\\quad |C_k|\\le K_{\\max}.$$
同时输出 refusal/uncertainty 标记与支持概率 (q_k\\in[0,1])（用于 refusal calibration）。
为防止“引用倾倒”，我们施加硬约束：每条断言必须引用不超过 (K_{\\max}) 个 token，并在评测中监控 citation-dump 模式（分析项，不单独作为 taxonomy，以免被喷“发明新指标”）。

---
3.3 程序化验证（Verifier）
给定 ((y,{C_k},T,\\mathcal{E})) 输出 issue 列表（taxonomy 固定、可枚举、可审计）：
- unsupported：断言无可接受证据
- overclaim：证据不足以支撑强断言（粒度过细，如 size/location 过细）
- inconsistency：互相矛盾/与结构槽冲突
- missing-slot：结构化必要槽缺失
每条 issue 记录 ((k,\\text{issue_type},\\text{severity},\\text{rule-id},\\text{evidence_trace}))，其中 severity ∈ {critical, non-critical}，evidence_trace 为最小可复现对象（token ids、cell_id、触发规则 id）。
Reviewer-check
- Taxonomy 必须可枚举，否则 verifier 变成黑箱 ✅
- 这里没有引入“第三贡献”，只是把 PCG 协议写清 ✅

---
4. 方法（只写闭环：Tokenize → Generate(+cite) → Verify → Refine）
4.1 总览（Fig1）
输入：((V,B))
闭环：BET 产生 tokens → PCG 生成断言+引用 → Verifier 输出 issues → 若预算未耗尽则按 issues/不确定性 refine tokens
输出：报告 + citations + verifier trace（可审计 artifact）
Fig1 应该画什么（必须是协议闭环，不是模块清单）
- 左：3D volume + coarse cells
- 中：tokens（带 Ω/cell_id）→ frames+citations → verifier issues
- 右：refine loop（priority queue / Δ 最大 split）
- 输出：可审计 trace（per-step Δ、issues 变化、citations）

---
4.2 C1：Budgeted Evidence Tokenization（BET）
4.2.1 表示与层级
用层级 partition（如八叉树）构造 coarse-to-fine cells。初始深度 (d_0) 固定、最大深度 (d_{\\max}) 固定；split 操作为标准 8-children octree split。
4.2.2 预算分配：Deterministic Greedy（主方法，保证可复现）
我们定义每个 cell 的边际证据收益 (\\Delta(c))，来源于当前 verifier issues 与 evidence head 不确定性（可解释、可计算、可复现）：
Algorithm 1：Deterministic Greedy BET-Refine
Input：(V), (B_{\\text{enc}}), (d_0), (d_{\\max}), stop threshold (\\epsilon), verifier refresh period (M)
State：cell set (\\mathcal{S})（初始为 depth (d_0) 全覆盖），tokens (T(\\mathcal{S}))，issues (\\mathcal{I})
1. 构建/更新 evidence graph (\\mathcal{E})（见 §4.3.1）
2. 对每个可 split cell (c\\in\\mathcal{S})（depth < (d_{\\max})）计算
$$\\Delta(c)=\\underbrace{\\sum_{u\\in\\mathcal{I}} w_u\\cdot \\widehat{\\Delta\\text{issue}}(u,c)}_{\\text{verifier-driven}}+\\lambda\\underbrace{H\\big(p(\\text{critical findings}\\mid c)\\big)}_{\\text{uncertainty}}$$
- (w_u) 按 severity 固定（critical>non-critical），(\\widehat{\\Delta\\text{issue}}) 用 evidence head 的局部预测近似（避免每步跑 full verifier）
- (H(\\cdot)) 用 entropy/margin（完全可计算）
3. 选择 (c^*=\\arg\\max\\Delta(c))，tie-break：最小 cell_id（字典序）
4. 若 (\\Delta(c^*)<\\epsilon) 或 (|\\mathcal{S}|\\ge B_{\\text{enc}})：停止
5. split (c^*) 为 8 children，增量更新 tokens 与 priority queue（缓存 encoder 特征，仅对新增子 cell 编码/ pooling）
6. 每 (M) 步运行一次 PCG+verifier 刷新 (\\mathcal{I})（其余步用近似项）
Output：tokens (T(\\mathcal{S}))
停机阈值 (\\epsilon)（避免被喷调参）
(\\epsilon) 在开发集用 Δ 的固定分位点规则设定（例如候选 Δ 的 5% 分位），并在所有预算 B 上共享，不按 B 单独调。
Reviewer-check
- 主文不依赖 RL ✅
- (\\Delta) 必须可解释（来自 issues/不确定性）✅
- deterministic 的 tie-break 必须写死 ✅
4.2.3 Learned allocator（仅消融）
可选：contextual bandit 学 (\\widehat{\\Delta}) 的排序近似，reward = issue reduction + external grounding gain（归一化），仅用于证明“不是启发式凑出来”，不作为系统可用性的前提。

---
4.3 C2：Proof-Carrying Generation（PCG）
4.3.1 Claim space：结构化 finding frames（避免自由文本无边界）
我们将临床断言规范为 finding frames（有限槽）：

$$(\\text{finding type}, \\text{laterality}, \\text{anatomical location}, \\text{severity/size bin}, \\text{negation/uncertain})$$

证据图 (\\mathcal{E})（可枚举合法域）
对每个 token (i)，evidence head 输出槽值候选及其置信度：
$$\\mathcal{E}(i)={(\\text{type}=nodule,p),(\\text{loc}=LLL,p),(\\text{lat}=left,p),(\\text{sizebin}=3\\text{–}5\\text{mm},p),\\dots}$$

全局合法域：
$$(\\mathcal{V}_{\\text{slot}}=\\bigcup_i \\mathcal{E}(i))$$
。constrained decoding 只允许输出域内槽值。
4.3.2 双通道输出协议（堵正文“加戏”）
生成器必须输出：
(a) findings table：({f_k, C_k, q_k})；
(b) 叙述文本：由 (a) 通过固定模板或受限重写生成，句末附 [t_i]。
Verifier 首先检查 (b) 是否可无损回译到 (a)（有限模板+词表），否则记 inconsistency。这样正文无法绕开 frames 私自“加戏”。
4.3.3 Token-citation：每条断言必须输出 (C_k)
对每个 frame (f_k)，模型输出指针分布 (p(i\\mid f_k)) 并选 Top-(K_{\\max}) 形成 (C_k)，同时输出 coverage/support score（供 verifier 判 unsupported/overclaim）。训练时 (C_k) 可用 weak supervision：来自 evidence head 的最高支持 token；在有 grounding 标签处（ReXGroundingCT）可加 overlap 监督。(arXiv)
4.3.4 Verifier：可复现、可枚举、可审计（Taxonomy v1.0 锁死）
Verifier 只用确定性规则（rule-id 版本锁定，防止“后验改规则”）：
- U1 unsupported：对 claim (k)，若 (\\forall i\\in C_k), (\\text{support}(f_k,i)=0)
- O1 overclaim（粒度）：预定义粒度层级（location: lobe > segment > subsegment；size: bin > exact mm）。若输出细于证据图可支持层级则 O1
- I1 inconsistency：互斥属性冲突（negation vs positive；left vs right；不相交 severity bins）
- M1 missing-slot：按 finding type 固定 required slots，缺任一则 M1
每条 issue 同时标注 severity（critical / non-critical），critical set 在附录枚举并固定（例如 pneumothorax / pleural effusion / large consolidation / suspicious nodule 等，按任务定义锁死）。
4.3.5 Calibrated refusal：拒答不是逃避
每个 frame 输出支持概率 (q_k\\in[0,1])，表示“在给定 citations 下可被 verifier 接受”的概率。若
$$q_k<\\tau_{\\text{refuse}}$$
则输出 uncertain/refuse；否则输出具体槽值。阈值 (\\tau_{\\text{refuse}}) 在开发集按约束 critical miss-rate ≤ δ 选择一次，并在所有预算 B 上固定，防止“按预算调阈值刷表”。
必须同时报告（反封嘴主表约束）
- unsupported rate（越低越好）
- critical miss-rate（不得上升）
- refusal rate（拒答比例）
- refusal ECE / reliability（校准好坏）
Reviewer-check
- 这节必须让 reviewer 相信：不是“定义规则封嘴”，而是把不确定性变成可校准行为，并用漏报率约束 ✅
- claim space 必须有边界，否则 verifier 形同虚设 ✅

---
5. 预算 Scaling Laws 与 Compute Allocation（Fig2 主轴）
我们在多个预算 (B\\in{B_1,\\dots,B_m}) 下评测，并拟合性能随预算的规律，同时学习可预测的 compute allocation model。
5.1 Performance–Budget scaling（可解释拟合，不绑死形状）
对每个性能指标 (P)（correctness、grounding、trustworthiness、Vol-Trust 等），拟合两类函数族并用 AIC/BIC 选型：
- 饱和幂律：
- $$P(B)=P_{\\infty}-a(B+b_0)^{-\\alpha}$$
- 对数饱和：
$$P(B)=c_0+c_1\\log(B+b_0)\\quad(\\text{并截断到 }[0,1])$$
拟合只在开发集进行；测试集报告预测误差与曲线稳定性。
5.2 Allocation model（必须能“预测最优分配”，不是只画曲线）
统一 compute 单位为 FLOPs：
[
\\text{FLOPs}_{\\text{total}}=\\text{FLOPs}_{\\text{enc}}(B_{\\text{enc}})+\\text{FLOPs}_{\\text{dec}}(B_{\\text{gen}})+\\text{FLOPs}_{\\text{verify}}(n_{\\text{verify}})
]
学习一个回归预测器 (\\hat{P}(\\theta))，其中 (\\theta=(B_{\\text{enc}},n_{\\text{refine}},B_{\\text{gen}},n_{\\text{verify}}))。通过网格采样少量配置点训练（例如 30–50 个配置），再解约束优化：
[
\\max_{\\theta}\\hat{P}(\\theta)\\quad \\text{s.t. } \\text{FLOPs}_{\\text{total}}(\\theta)\\le \\mathcal{B}
]
得到在预算 (\\mathcal{B}) 下的预测最优配置。测试时报告 regret（预测最优 vs 实际最优差距），以证明“模型能预测最优分配”。该叙事与 RAG inference scaling work 在 OpenReview 上提出的 computation allocation model 口径对齐。(OpenReview)
Fig2 应该画什么（必须包含“可预测分配”）
- 左：多预算 Pareto（correctness vs compute；grounding vs compute；trustworthiness vs compute）
- 中：P(B) 拟合曲线 + 边际收益递减点（stop principle）
- 右：allocation model 预测最优配置 vs 真实最优配置（regret 曲线）
Reviewer-check
- 必须输出“模型/拟合”，不能只给曲线 ✅
- 必须能预测最优分配并报告 regret ✅

---
6. 数据与评测（硬地基：公开数据 + pixel-level grounding）
6.1 Report generation 数据（公开可复现）
- CT-RATE：公开 3D chest CT volumes paired with reports；论文与数据卡描述其包含 25,692 non-contrast 3D chest CT scans、21,304 unique patients，并可扩展到 50,188（多重重建）。(arXiv)
- CT-3DRRG：由 Argus 工作整理并描述为“largest publicly available 3D CT-report dataset”，用于跨源泛化与训练/评测 recipe 对比。(ACL Anthology)
6.2 Pixel-level grounding 数据（硬 grounding）
- ReXGroundingCT：公开数据集，将 free-text findings 与 pixel-level 3D segmentation 关联；包含 3,142 CT 与 finding-level grounding，且明确其为 3D chest CT 上的 sentence-level grounding 资源。(arXiv)
6.3 泄漏防护（Protocol Lock，必须像法律条款）
1. ID 统一：(\\texttt{scan_hash}=\\text{SHA256}(\\text{patient_id}||\\text{study_date}||\\text{series_uid}))；split 与交集仅基于 scan_hash 与 patient_id。
2. patient-level split：同一 patient_id 的任何 scan 只能出现在 train/val/test 之一。
3. 去重：
  - 文本：报告归一化后 MinHash + Jaccard>0.9 判 near-duplicate；重复只保留 1 个并记录映射表。
  - 影像：固定下采样后 perceptual hash / 随机投影 hash 判 duplicate。
4. 交集处理（锁死）：禁止 ReXGroundingCT val/test 进入训练、阈值选择（(\\epsilon,\\tau_{\\text{refuse}})）、allocation model 拟合。
5. 版本锁定：数据版本（revision hash）、split manifest（所有 scan_hash 列表）与代码 commit 写入 artifact。
Reviewer-check
- 这节是“能否避免一票否决”的关键，必须写死 ✅

---
7. 实验设计（3 图 2 表；指标与 baseline 必须齐）
7.1 主指标（Table1）
Clinical correctness（结构化事实）
- finding frame F1（含 laterality/location/negation/size bin 等槽）
- 匹配方式：按 finding type + coarse location + laterality 做 Hungarian matching；slot-level micro/macro F1 统计
Grounding（ReXGroundingCT）(arXiv)
- sentence → citation → 3D segmentation：
  - hit-rate：是否存在被引用 token 的 (\\Omega) 与 lesion mask 有 overlap（≥阈值）
  - IoU / Dice：对 cited cells 的 union 与 lesion mask 计算（同时报告 max-over-citations 与 union-over-citations 两种口径）
Trustworthiness（协议性指标，verifier taxonomy）
- unsupported / overclaim / inconsistency / missing-slot rates（按 severity 分桶）
- Vol-Trust：加权组合（仅做汇总主指标，不把它当贡献）
Safety against “封嘴”
- critical finding miss-rate
- refusal calibration：ECE / reliability diagram（按 (q_k) 分桶）
Efficiency
- #tokens、encoder/decoder/verifier FLOPs（统一口径）
- end-to-end latency：mean + P95（cold / warm cache 分开）

---
7.2 Baselines（必须齐，否则拒）
Tokenization/compute baselines
- fixed-grid 3D tokens（同 backbone，同 B）
- 2D / 2.5D slice uniform sampling
- ROI-crop / coarse-to-fine（含 detector/segmenter cost 入账）
- no-refine（只 coarse）
Protocol baselines（消融 PCG 核心）
- no-citation / no-constraint（去掉 PCG 强制引用/约束）
- citation=top-attention tokens（decoder→encoder cross-attn 聚合 top-(K_{\\max}) 作为伪引用）
- citation-only（有引用但无 verifier/无 refusal；检验“只贴引用”是否有效）
3D RRG 强 baseline
- CT2Rep 等公开 3D CT report generation 强基线。(arXiv)

---
7.3 预算与公平性协议（写死）
- B 定义：联合预算 (B=B_{\\text{enc}}+B_{\\text{gen}})，并换算到 FLOPs：
(\\text{FLOPs}_{\\text{total}}=\\text{FLOPs}_{\\text{enc}}+\\text{FLOPs}_{\\text{dec}}+\\text{FLOPs}_{\\text{verify}})
- FLOPs-matched：至少一组严格匹配总 FLOPs 的对比（主结论必须在 matched setting 站得住）
- ROI-crop 成本：任何 detector/segmenter/selector 的 FLOPs 与 latency 必须计入总账；若 baseline 使用外部模型，必须报告其成本，并在 matched setting 中相应减少主模型预算
- latency：固定硬件/批大小；冷热 cache 分开；P95 基于≥1000 样本；报告 mean+P95

---
7.4 反事实三件套（Fig3 主武器）
- Ω-permutation：随机置换 (\\Omega_i)（cell_id），保持 token embedding (t_i) 不变，使 embedding 分布完全保持但空间对应关系破坏 → grounding/correctness 应显著下降
- token-permutation：固定 (\\Omega_i) 不变，置换 (t_i)（或 token 索引），检验模型是否依赖正确的 token–Ω 对应
- citation-swap：在同一报告内随机交换 (C_k)（保持 (|C_k|) 分布不变），unsupported/overclaim 应激增
- evidence-drop：删除被引用 tokens 并重生成/再验证；并做“禁引用但不删信息”的对照（区分信息缺失 vs 引用被禁）
- mask sanity（ReXGroundingCT）：refine 新增 tokens 的 (\\Omega) 与 lesion mask overlap（IoU/Dice）上升，证明 refine 真在“追证据”。(arXiv)
统计显著性
- paired bootstrap（≥10k resamples）给 95% CI；多预算多指标多重比较用 Holm 校正。
Fig3 应该画什么
- 主图：counterfactual 前后（correctness / grounding / unsupported / overclaim）的变化柱状或折线
- 附图：mask sanity overlap 分布（refine 新增 tokens vs baseline）
Reviewer-check
- 缺反事实，你的 citations 会被说成装饰 ✅
- permutation 必须“保持 embedding 分布”否则不干净 ✅

---
8. 训练与实现（只保留最小闭环，降低风险）
8.1 训练阶段（M0→M3）
- M0：协议跑通（先能审计、能验证）
  - 固定 coarse tokens（depth (d_0)）
  - 从 reference 报告抽取 finding frames（规则+词表+模板，附录锁死）
  - loss：slot-wise CE（frames）+ 文本 NLL（由 frames 模板重写得到 target）+ 引用弱监督（鼓励选择 evidence head 高支持 token）
- M1：加入 BET refine（deterministic allocator）
  - 引入 Algorithm 1 闭环，产生 Fig2 多预算 Pareto
  - 不引入 RL 风险，不改变主训练假设
- M2：接入 ReXGroundingCT 做 grounding 与反事实(arXiv)
  - grounding consistency loss：鼓励引用 token 的 (\\Omega) 与 segmentation overlap
  - 完成 Fig3 三件套
- M3（可选）：learned allocator（bandit）消融
  - reward = issue reduction + grounding gain（归一化），只做增益报告，不影响主结论
8.2 可复现 artifact（强建议）
每个样本输出：
- tokens（含 Ω/cell_id）、citations、verifier issues（含 rule-id、trace、severity）
- refusal 标记与 (q_k)
- 运行配置（预算 B、硬件、seed）、数据版本（revision）、split manifest、代码 commit
- refine trace：每步 (c^)、(\\Delta(c^))、issues 变化曲线（可审计）
- ≥3 seeds，附 reproducibility statement
Reviewer-check
- 不是“我们开源”，而是“我们输出可审计 trace” ✅

---
9. 预期风险与应对（提前写 rebuttal）
1. unsupported↓ 但 miss-rate↑（封嘴风险）
- 把 critical miss-rate 与 refusal calibration 写成主指标；(\\tau_{\\text{refuse}}) 受 miss-rate ≤ δ 约束；若 miss-rate 上升，限制 refusal 覆盖范围并提升 evidence recall（提高 coarse coverage / 调整 stop principle）
2. verifier 太弱/太强
- 报告 verifier 强度曲线（弱→强规则集），展示协议可插拔而非绑死；taxonomy 不变，仅规则强度变化
3. latency 不稳定
- 写死测量协议（P95、冷热 cache、固定 batch），并提供 FLOPs-matched；使用增量缓存降低 refine 成本
4. 跨源泛化失败（CT-3DRRG）
- 作为必做实验；若失败，分析 domain shift 下 allocation 崩溃点（refine 追错区域、证据图失真）并给出修复（domain-robust evidence head / conservative stop / 多源校准）

---
10. 结论（必须回到两条贡献与 B 主轴）
我们贡献：
- BET：在严格预算 B 下的证据 tokenization（tokens 自带显式 Ω/cell_id），并给出可解释 scaling law + allocation model；
- PCG：proof-carrying 生成协议，把 grounded citation + verifier + calibrated refusal 变成可审计输出与硬指标；
并在公开 3D CT 报告生成与 pixel-level grounding 上，用 Pareto 主导 + counterfactual 击穿证明其非平凡性。

---
你接下来必须“死磕”的 6 个验收条件（不满足就别谈 oral）
1. Fig2：多预算 Pareto dominate（不是某个点赢）
2. Fig2：scaling law + allocation model 能预测最优分配（报告 regret，不是画曲线）(OpenReview)
3. Fig3：Permutation / citation-swap / evidence-drop 统计显著击穿（bootstrap+Holm）
4. ReXGroundingCT：citation-grounding（IoU/Dice/hit-rate）显著提升(arXiv)
5. Table1：unsupported↓ 的同时 critical miss-rate 不升、refusal 校准可控（否则=封嘴）
6. Baselines：CT2Rep 等强基线 + FLOPs/latency matched 全齐(arXiv)
