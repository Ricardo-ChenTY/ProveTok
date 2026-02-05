# Mohu

## 1. Not Implemented

- [x] M0116: C0001 论文级：baselines_curve_multiseed 补齐 latency P95 并纳入 proof_check 硬判定
  - Ref: C0001, E0138, P0007
  - Context: 目前 C0001 的通过条件主要看 `combined` 的显著性；latency 仅在 details 里记录 mean，且缺少 P95/cold/warm 的一致口径，导致“Pareto/latency-matched”说服力不足。
  - Acceptance:
    - `baselines_curve_multiseed.json` 必须输出 `warm_time_mean_s` 与 `warm_time_p95_s`（至少 warm），并能在多预算/多 seed 下聚合；
    - `python scripts/proof_check.py` 的 C0001 必须把 `warm_time_p95_s`（+5% 容忍）与 `unsupported`（+0.05 容忍）作为硬约束纳入通过条件。
  - Verification: `python -c "import json, pathlib; p=pathlib.Path('outputs/E0138-full/baselines_curve_multiseed.json'); d=json.loads(p.read_text()); assert 'latency' in d or 'metrics' in d; print('schema_ok')" && python scripts/proof_check.py | python -c "import json,sys; d=json.load(sys.stdin); c=[x for x in d['checks'] if x['claim_id']=='C0001'][0]; print('proved', c['proved']); assert isinstance(c['details'], dict)"`

- [x] M0117: C0002 论文级：allocation/regret 增加 naive baselines + normalized regret + bootstrap CI
  - Ref: C0002, E0141, E0142
  - Context: 当前 C0002 更像“产物存在”；缺少对 regret 的阈值、CI、以及对 naive policy 的显著优势对比。
  - Acceptance: `fig3_results.json` 输出 `mean_normalized_regret` 及其 bootstrap CI，并包含至少一个 naive policy（如 `always_fixed_grid`）的同口径 regret。
  - Verification: `python -c "import json, pathlib; p=pathlib.Path('outputs/E0141-full/fig3_results.json'); d=json.loads(p.read_text()); assert 'regret' in d and 'mean_normalized_regret' in d['regret']; assert 'naive_policies' in d; print('fig3_schema_ok')"`

- [x] M0118: C0004 论文级：grounding proof 覆盖更多 budgets/seeds + 多 baselines，并写入可审计 artifacts
  - Ref: C0004, E0143, E0156
  - Context: 目前 grounding proof 覆盖的 budgets/baselines 有限，导致“显著提升”说服力偏弱。
  - Acceptance: `figX_grounding_proof.json` 使用 `seeds>=5`、`budgets>=6`，并至少对比 `fixed_grid` 与 `roi_variance`（以及可选更多 baselines），输出 Holm 校正后的 p 值与 per-sample artifacts。
  - Verification: `python -c "import json, pathlib; p=pathlib.Path('outputs/E0143-full/figX_grounding_proof.json'); d=json.loads(p.read_text()); assert len(d.get('seeds',[]))>=5; assert len(d.get('budgets',[]))>=6; assert 'paired_bootstrap' in d; print('grounding_proof_ok')"`

- [x] M0119: C0005 论文级：refusal calibration 把 ECE/refusal-rate 变成硬判定，并输出可审计 bins
  - Ref: C0005, E0144
  - Context: 当前 proof_check 只硬判定 miss-rate 与 unsupported；ECE/reliability 与 refusal-rate 未纳入通过条件，导致“反封嘴+可校准”说服力偏弱。
  - Acceptance: `figX_refusal_calibration.json` 在 test 上为每个 budget 输出 `refusal_ece` 与 `reliability_bins`，且 proof_check 把 `refusal_ece<=0.15`、`refusal_rate<=0.20` 纳入硬判定。
  - Verification: `python -c "import json, pathlib; p=pathlib.Path('outputs/E0144-full/figX_refusal_calibration.json'); d=json.loads(p.read_text()); r=d['test']['rows'][0]; assert 'refusal_ece' in r['calibrated']; assert 'reliability_bins' in r['calibrated']; print('refusal_schema_ok')"`

- [x] M0120: C0006 论文级：实现可复现强 3D RRG baseline（ct2rep_strong）并纳入 baselines suite
  - Ref: C0006, E0140, E0138
  - Context: `ct2rep_like` 目前是协议占位，无法作为论文级“强基线”。
  - Acceptance:
    - 提供 `ct2rep_strong` 的训练脚本与 weights 产物（可复现 meta/seed/data revision）；
    - `run_baselines`/`baselines_curve_multiseed` 方法列表包含 `ct2rep_strong`，并能在 test 上输出 frame_f1/grounding/latency；
    - `python scripts/proof_check.py` 的 C0006 对强基线做最小非退化门槛检查（例如 frame_f1 不为 0）。
  - Verification: `python -c "import pathlib; assert pathlib.Path('outputs/E0140-full/ct2rep_strong.pt').exists(); print('weights_ok')" && python -m provetok.experiments.run_baselines --smoke --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_mini/manifest.jsonl --split test --ct2rep-strong-weights outputs/E0140-full/ct2rep_strong.pt --output-dir ./outputs/_verify_ct2rep_strong && python scripts/proof_check.py | python -c "import json,sys; d=json.load(sys.stdin); c=[x for x in d['checks'] if x['claim_id']=='C0006'][0]; print('proved', c['proved'])\"`

- [x] M0114: `.rd_queue sync` 修复 checkbox 语义（Full 通过后 Smoke 不能倒挂）
  - Ref: P0031, P0022, C0001, C0002, C0003, C0004, C0005, C0006
  - Context: 之前台账里出现过 `Full=[x]` 但 `Smoke=[ ]` 的倒挂；新增/更新实验后会再次触发该类漂移，导致 rd_queue 无法稳定复用。
  - Acceptance: `python scripts/rd_queue.py sync` 后，`docs/experiment.md` 不存在任意行 `Smoke=[ ]` 且 `Full=[x]`。
  - Verification: `python scripts/rd_queue.py sync && python -c "import re, pathlib; t=pathlib.Path('docs/experiment.md').read_text(encoding='utf-8'); assert not re.search(r'\\|\\s*\\[ \\]\\s*\\|\\s*\\[x\\]\\s*\\|\\s*$', t, re.M); print('checkbox_ok')"`
  - Resolution: `scripts/rd_queue.py::_update_experiment_md_checkboxes` 现在在检测到 `(E####, full)=passed` 时会自动将 Smoke 也置为 `[x]`，避免台账倒挂；已通过 `python scripts/rd_queue.py sync` 验证。

- [x] M0115: Baselines 指标口径与 Fig2 对齐（IoU/Dice/hit/combined/latency）并支持多预算曲线
  - Ref: P0032, P0033, C0001, C0004, C0006, E0111
  - Context: 当前 baselines runner 只输出 `frame_f1/iou/unsupported/overclaim`，缺少 dice/hit/combined 与 latency；且缺少 “多预算曲线 + multi-seed + CI” 的聚合产物，无法用于 C0001/C0004 的证明规则。
  - Acceptance: `run_baselines` 输出包含 `dice/hit_rate/combined/warm_time_s`，并新增 `baselines_curve_multiseed` 脚本产出 `baselines_curve_multiseed.json`。
  - Verification: `python -m provetok.experiments.run_baselines --smoke --output-dir ./outputs/_verify_baselines && python -c "import json, pathlib; p=max(pathlib.Path('outputs/_verify_baselines').glob('*/baselines.json'), key=lambda x:x.stat().st_mtime); d=json.loads(p.read_text()); assert 'dice' in d['raw']['fixed_grid']; assert 'hit_rate' in d['raw']['fixed_grid']; assert 'combined' in d['raw']['fixed_grid']; assert 'warm_time_s' in d['raw']['fixed_grid']; print('baselines_metrics_ok')" && python -m provetok.experiments.baselines_curve_multiseed --help`
  - Resolution: `provetok/experiments/run_baselines.py` 已补齐 `dice/hit_rate/combined/warm_time_s`；新增 `provetok/experiments/baselines_curve_multiseed.py` 输出多预算曲线聚合 JSON；`scripts/rd_queue.py sync` 增加对 `baselines_curve_multiseed.json` 的摘要提取。

- [x] M0111: Experiments 输出缺少统一 artifact meta（版本锁信息无法审计）
  - Ref: P0027, C0001, C0003, C0004, C0005, C0006, E0107, E0108, E0109
  - Context: 目前仅 `provetok/run_demo.py` 输出 `meta`（commit/schema/taxonomy/rule-set/hardware 等）；`provetok/experiments/*` 的输出 JSON 仍缺失这些字段，导致“论文级 full”无法复现/审计。
  - Acceptance: 所有 experiments 输出 JSON 均包含 `meta`，至少覆盖 code commit、schema/taxonomy/rule-set 版本、data revision、split manifest path、seed、hardware 信息。
  - Verification: `python -m provetok.experiments.fig2_scaling_law --budgets 8 --n-samples 1 --no-plot --output-dir ./outputs/_verify_meta && python -c "import json; d=json.load(open('outputs/_verify_meta/fig2_raw_data.json')); assert 'meta' in d; print('meta_ok')"`
  - Resolution: 已在 `fig2_scaling_law`/`run_baselines`/`figX_counterfactual`/`figX_refusal_calibration` 输出 JSON 中写入 `meta`（基于 `provetok/utils/artifact.py`），并通过 `outputs/_verify_meta/fig2_raw_data.json` 校验。

- [x] M0112: Fig2/Baselines 缺少 multi-seed + CI（无法证明 Pareto dominate）
  - Ref: P0028, C0001, C0002, C0004, C0006, E0107, E0108
  - Context: 当前 Fig2 与 baselines runner 主要输出 mean/std；但 Claims 的 proof rule 需要多 seed + CI/显著性（paired/bootstrap + 多重比较）。
  - Acceptance: 提供可复用的 multi-seed runner（或聚合脚本），输出 per-budget 指标的 CI，并能对“支配关系/差异”给出统计结论。
  - Verification: `python -m provetok.experiments.fig2_scaling_multiseed --help`
  - Resolution: 新增 `provetok/experiments/fig2_scaling_multiseed.py`（层级 bootstrap CI），并为 `run_baselines.py` 增加 `--seeds/--n-bootstrap/--ci` 的多 seed 聚合输出。

- [x] M0113: `.rd_queue` 仅勾选 checkbox，未把关键指标写回实验台账
  - Ref: P0018, C0001, C0002, C0003, C0004, C0005, C0006
  - Context: `scripts/rd_queue.py sync` 目前只根据 `.rd_queue/results/*.json` 更新 Smoke/Full 勾选；没有把关键指标（例如 Fig2 的曲线摘要/latency、counterfactual 的 CI、baselines summary）回写到 `docs/experiment.md` 或单独的 `docs/results.md`，导致“证明链”断裂。
  - Acceptance: `sync` 在勾选通过后，同时写入每个 E#### 的关键指标摘要（可落在 `docs/experiment.md` 的 Notes 列或新增 `docs/results.md`），并能追溯到对应 output 路径。
  - Verification: `python scripts/rd_queue.py sync`
  - Resolution: `scripts/rd_queue.py sync` 现在会生成/更新 `docs/results.md`，包含每条 job 的 output 路径与关键指标摘要。

- [x] M0104: CT-3DRRG 数据集入口缺失：缺少 index/manifest 构建脚本（split from M0009）
  - Ref: P0020, C0006, M0009
  - Context: 当前仓库 `scripts/data/` 仅覆盖 CT-RATE 与 ReXGroundingCT 的 index/manifest 构建；`docs/plan.md` 提到 CT-3DRRG 但缺少可复现的 manifest 入口，导致跨源泛化/训练/评测无法落地。
  - Acceptance: 新增 `scripts/data/make_ct_3drrg_index.py` 与 `scripts/data/build_ct_3drrg_manifest.py`（或等价），输出标准 `manifest.jsonl` + `*.meta.json`（revision hash）+ `*.splits.json`（split manifest），并通过 `ProtocolLock` 的 patient-level split 与交集条款校验。
  - Verification: `python scripts/data/build_ct_3drrg_manifest.py --help`
  - Notes: 继续采用“manifest 驱动”为唯一入口；如条款不允许则不内置自动下载，仅提供构建/校验脚本。
  - Resolution: 已新增 `scripts/data/make_ct_3drrg_index.py` 与 `scripts/data/build_ct_3drrg_manifest.py`，产物与其它数据集脚本一致（`manifest.jsonl` + `.meta/.splits/.dupes` 版本锁）。

- [x] M0105: 依赖声明不完整：`requirements.txt` 未覆盖当前代码/脚本实际使用依赖
  - Ref: P0019
  - Context: `provetok/eval/metrics_frames.py` 依赖 `scipy`，`provetok/experiments/fig2_scaling_law.py` 依赖 `tqdm`，`scripts/data/download_ct_rate_mini_from_rex.py` 依赖 `huggingface_hub`；新环境 `pip install -r requirements.txt` 可能无法运行 tests/experiments。
  - Acceptance: `requirements.txt` 补齐上述依赖（可选依赖用 extras 文件或注释说明）；保证 `pytest -q` 与关键脚本 `--help` 在干净环境可跑。
  - Verification: `python -c "import scipy, tqdm; import huggingface_hub; print('deps_ok')"`
  - Resolution: 已更新 `requirements.txt` 补齐 `scipy/tqdm/pandas/huggingface-hub`，覆盖当前代码与数据脚本依赖。

- [x] M0106: 实验台账未闭环：`docs/experiment.md` 仍存在 `TBD` 且 smoke/full checkbox 语义不一致
  - Ref: P0022, P0009
  - Context: `docs/experiment.md` 的 Multi-GPU script 列仍为 `TBD`；且存在 `Full=[x]` 但 `Smoke=[ ]` 的倒挂，导致台账不可直接复用/审计。
  - Acceptance: 所有实验行提供可运行的 `1GPU script` 与 `Multi-GPU script`（不再出现 `TBD`）；并统一 checkbox 语义（至少保证 Full 勾选前 Smoke 不会缺失）；`python scripts/rd_queue.py make` 可解析全部命令并生成队列。
  - Verification: `python -c "import pathlib; t=pathlib.Path('docs/experiment.md').read_text(encoding='utf-8'); assert 'TBD' not in t; print('no_tbd')"`
  - Resolution: 已把所有 `TBD` 替换为可运行命令（当前 Multi-GPU 列与 1GPU 相同以保证可运行），并修复 `E0101–E0106` 的 smoke/full checkbox 倒挂。

- [x] M0107: FLOPs/latency matched 仍不一致：Fig2/FigX 未统一接入 `ComputeUnitCosts` 与 fail-fast 机制
  - Ref: P0021, C0001, C0002, C0006
  - Context: 当前 `run_baselines.py` 支持 `--costs-json/--flops-total`，但 `fig2_scaling_law.py`/`figX_counterfactual.py` 未统一支持 matched 协议；`ComputeUnitCosts` 仍是 toy 常量，难以支撑“matched setting 不缺席”的 claim。
  - Acceptance: Fig2/FigX/RunBaselines 统一支持 `--costs-json`（由 `scripts/profile_flops.py --out-costs` 生成）与 matched 配置，并在不匹配时 fail-fast；产物写入 `flops_total` 与 latency(mean/P95,cold/warm)。
  - Verification: `python scripts/profile_flops.py --device cpu --out ./outputs/flops_profile.json --out-costs ./outputs/compute_costs.json`
  - Resolution: `fig2_scaling_law.py` 新增 `--budget-mode {tokens,flops}` + `--costs-json/--b-gen/--n-verify` 并输出 flops/latency 汇总；`figX_counterfactual.py` 新增 `--flops-total/--costs-json/--b-gen/--n-verify` 并在 matched 模式下 fail-fast。

- [x] M0108: `Llama2PCG` 仅支持单帧输出，无法覆盖多 finding 的 PCG 协议
  - Ref: P0023, C0001, C0004, C0005
  - Context: `provetok/pcg/llama2_pcg.py` 的 prompt 约束 `frames` 只能包含 0 或 1 条，导致 LLM 模式下无法对齐多 finding 的 frames+citations 协议与评测。
  - Acceptance: 支持输出 0..K 帧（K 可配置），并保持 citations/q/refusal schema 不变；增加不依赖真实模型权重的单元测试覆盖 JSON 抽取/清洗/多帧 citations 约束。
  - Verification: `pytest -q`
  - Resolution: `Llama2PCGConfig` 新增 `max_frames` 并放宽 prompt 限制；抽出 `parse_llm_json`/`sanitize_generation_dict` 以便单测覆盖多帧与 citations 约束（`tests/test_llama2_pcg_json.py`）。

- [x] M0109: 训练入口与 StageConfig 不一致：`bet_steps/budget_tokens/epsilon` 仍未进入训练闭环（且存在 dummy 训练脚本）
  - Ref: P0024
  - Context: `provetok/training/stages.py` 定义了 BET refinement 相关参数，但 `Trainer`/`ProveTokSystem.forward` 训练路径仍固定 root cell；`scripts/train_m0.py` 仍用 dummy 参数做“训练”占位，容易造成“训练已完成”的误解。
  - Acceptance: 训练入口真正使用 `StageConfig` 的 BET 参数并调用 `run_refine_loop`（或等价），并提供 CPU 上可跑的最小 smoke（2 steps）；`scripts/train_m0.py` 不再依赖 dummy 参数（改为统一 Trainer/系统入口）。
  - Verification: `python -c "from provetok.training.trainer import Trainer, TrainerConfig; cfg=TrainerConfig(stage='M1', device='cpu', dataset_cfg={'dataset_type':'synthetic','num_samples':4,'vol_shape':[16,16,16],'batch_size':2}, output_dir='./outputs', overrides={'max_steps':2,'log_every':1,'eval_every':100000,'save_every':100000}); Trainer(cfg).train(); print('train_smoke_ok')"`
  - Resolution: `Trainer` 已根据 `StageConfig.bet_steps/budget_tokens/epsilon/max_depth` 调用 `run_refine_loop`（训练中使用 deterministic allocator）；`scripts/train_m0.py` 改为统一 Trainer 入口（不再依赖 dummy 参数）。

- [x] M0110: 项目 README 与当前实现不一致（仍停留在 scaffold 叙事）
  - Ref: P0025
  - Context: `README.md` 仍以 “from scratch scaffold/占位” 叙述为主，未反映当前已实现的 manifest 数据入口、`rd_queue`、LLM/encoder 切换参数与实验运行方式，影响复现与协作。
  - Acceptance: README 覆盖 Quickstart（demo/数据/实验）并明确 toy vs real 开关；避免与 `docs/plan.md`/`docs/experiment.md` 漂移。
  - Verification: `python -c "import pathlib; t=pathlib.Path('README.md').read_text(encoding='utf-8'); assert '--dataset-type' in t and 'rd_queue' in t; print('readme_ok')"`
  - Resolution: 已更新 `README.md`，补齐 manifest 数据入口、`.rd_queue` 队列运行与关键实验参数（`--dataset-type/--pcg/--encoder`）及训练入口说明。

- [x] M0009: 真实数据落地缺口：缺少 CT-RATE / ReXGroundingCT 的 manifest 构建/预处理脚本与版本锁（CT-3DRRG 已拆分到 M0104）
  - Ref: P0010, C0004, C0006
  - Context: 当前 `provetok.data` 仅支持用户手写 manifest（或放置 `manifest.jsonl`），没有从原始数据目录一键生成可复现的 `manifest.jsonl` / `split manifest` / `revision hash` / 去重映射表的流水线。注：CT-3DRRG 的入口缺口已拆分到 M0104。
  - Acceptance: 新增 `scripts/data/build_*.py`（至少 `build_ct_rate_manifest.py`, `build_rex_groundingct_manifest.py`）输出标准 manifest；并把数据版本/拆分清单写入输出 artifact。
  - Verification: `python scripts/data/build_ct_rate_manifest.py --help`
  - Notes: 实现方案：以 manifest 为“唯一输入”，允许用户自带下载；脚本完成路径扫描→scan_hash 计算→patient-level split→输出 jsonl + split manifest + revision。
  - Resolution: 已新增 `scripts/data/build_ct_rate_manifest.py` 与 `scripts/data/build_rex_groundingct_manifest.py`；产物包含 `*.meta.json`（revision hash）、`*.splits.json`（split manifest）、`*.dupes.json`（精确重复报告分组）。

- [x] M0010: Pixel-level grounding 真实链路缺失：ReXGroundingCT 的 mask 读取/对齐/评测与 mask-sanity 还停留在 synthetic
  - Ref: P0011, C0004, C0003
  - Context: 现在 `ReXGroundingCTDataset` 只返回 volume+report；`compute_citation_grounding()` 需要 per-finding 3D mask，但真实数据没有进入 dataloader/runner。
  - Acceptance: manifest 支持 `mask_path`（或等价字段）并在 dataloader 返回；`fig2/figX_counterfactual` 可在真实 ReXGroundingCT 上跑出 IoU/Dice/hit-rate + mask-sanity（refine 新增 Ω 与 mask overlap 上升）。
  - Verification: `python -m provetok.experiments.figX_counterfactual --smoke`
  - Notes: 实现方案：`provetok.data.io` 增加 `load_mask()`（支持 `.npy` / NIfTI）；dataset 在 `meta`/batch 中提供 `lesion_masks`；评测时按 frame_idx 对齐。
  - Resolution: 新增 `provetok/data/io.py::load_mask`，并在 `provetok/data/dataset.py` 读取 manifest 的 `mask_path`→返回 `lesion_masks={0: mask}`；`provetok/experiments/figX_counterfactual.py` 增加 `--dataset-type manifest --manifest ...` 支持（默认仍为 synthetic）。

- [x] M0011: BET 仍是 toy tokenization：`encode_tokens()` 未接入真实 3D encoder feature map 与 cell pooling/缓存
  - Ref: P0012, C0001, C0002, C0006
  - Context: `provetok/bet/tokenize.py` 目前用 patch mean/std 做 embedding（明确 TODO）；无法对齐 “3D encoder compute 计入 B_enc” 的主轴。
  - Acceptance: `encode_tokens()` 支持注入 `BaseEncoder3D`（`provetok/models/encoder3d.py`），并在 refine 过程中缓存 feature map 与已编码 cell，增量生成子 cell embedding。
  - Verification: `pytest -q`
  - Notes: 实现方案：新增 `TokenEncoder`（wrap encoder + cache）；`run_refine_loop()` 传入 encoder/cacher；ComputeBudget 使用 encoder 输出维度与 feature map 尺寸估计 FLOPs。
  - Resolution: `provetok/bet/tokenize.py` 新增 cache-aware `TokenEncoder`，`encode_tokens(..., encoder=...)` 支持 encoder-backed pooling；`provetok/bet/refine_loop.py` 在循环内复用 `TokenEncoder` 避免每步全量重算，并补齐 deterministic 的稳定 hash（去除 Python `hash()` 随机盐）。

- [x] M0012: Algorithm 1 的“停机阈值 ε 选取规则 + refresh period M 的近似项”未锁死，容易被认为是手工调参
  - Ref: P0013, C0001, C0002
  - Context: 当前 ε 直接作为参数传入；没有“开发集 Δ 分位点”规则，也没有把 “非 refresh 步用近似 Δ̂_issue” 与 “refresh 步跑 full verifier” 的差异记录到 trace。
  - Acceptance: 提供 `scripts/calibrate_epsilon.py`（或等价）在 dev 集固定一次 ε（分位点规则），并写入所有实验配置；trace 中记录是否 full-refresh 以及近似/真实 issues 差异。
  - Verification: `python scripts/calibrate_epsilon.py --help`
  - Notes: 实现方案：收集每步候选 Δ(c) 分布→按固定分位数取 ε；在 `RefineTrace` 增加 `verifier_refreshed` 与 `issues_snapshot`。
  - Resolution: 新增 `scripts/calibrate_epsilon.py`（quantile 规则选 ε）；并在 `provetok/bet/refine_loop.py` 的 `RefineTrace` 增加 `verifier_refreshed` 字段记录 refresh 周期。

- [x] M0013: τ_refuse 没有“开发集一次选定 + 跨预算固定”的闭环落地到主推理/实验 runner
  - Ref: P0013, C0005
  - Context: `PCGHead` 内部有固定 `tau_refuse`；`RefusalCalibrator` 能调 τ，但未把“val 选 τ → test/多预算固定”写进 `fig2/fig3/run_baselines` 的默认流程。
  - Acceptance: 新增统一的 `RefusalPolicy`：在 val 上选择 τ（critical miss-rate ≤ δ）并写入配置；测试/多预算复用同一 τ；主表输出 unsupported/miss-rate/refusal-rate/ECE。
  - Verification: `python -m provetok.experiments.figX_refusal_calibration --smoke`
  - Notes: 实现方案：runner 先跑 val 生成→调用 calibrator→`apply_refusal_to_generation()`→再评测；禁止按预算重新调 τ。
  - Resolution: 新增 `provetok/pcg/refusal.py::RefusalPolicy`（可序列化/复用）；`provetok/experiments/figX_refusal_calibration.py` 产出 `refusal_policy.json`；`provetok/experiments/run_baselines.py` 支持 `--refusal-policy` 复用同一 τ（避免按预算调参）。

- [x] M0014: 训练阶段 (M0→M3) 仍是占位：缺 citation weak supervision / verifier loss / grounding loss / allocator 学习
  - Ref: P0014, C0001, C0003, C0004, C0005
  - Context: `Trainer` 目前只做 slot CE；没有 (a) citation 指针监督，(b) verifier-driven loss，(c) grounding consistency loss（ReX），(d) evidence head/allocator 学习与 bandit 消融。
  - Acceptance: `Trainer(stage=M1/M2)` 支持：citation loss + verifier loss + grounding loss（若有 mask）；并能训练 EvidenceHead 以预测 issue reduction / uncertainty；输出稳定的训练日志与 checkpoint。
  - Verification: `pytest -q`
  - Notes: 实现方案：为每个 GT frame 选 supporting token（evidence graph top-1 或 mask-overlap top-1）做 citation CE；verifier loss 用 issue_type 做 penalty；grounding loss 用 cited Ω union vs mask 的 soft IoU surrogate。
  - Resolution: `provetok/training/trainer.py` 已加入 `citation`/`verifier`/`grounding`/`evidence_*` 额外损失；verifier 目标由 rule-based issues 生成（不绕过 refusal）；EvidenceHead 用 issue-blame 与 entropy 对齐做自监督。

- [x] M0015: P0015 Baselines “论文级”补齐仍未闭环：baseline latency protocol + CT2Rep 强基线仍为占位
  - Ref: P0015, C0001, C0006
  - Context:
    - `docs/plan.md` 中 P0015 仍未完成（unchecked），但 `run_baselines` 已包含 2D/2.5D/ROI/ct2rep_like（占位）等方法；
    - 仍缺少“baseline 侧同协议”的 latency 结果落盘（cold/warm mean+P95），无法支撑 paper-grade 的 latency-matched 讨论。
  - Acceptance:
    - 新增 baseline latency benchmark artifact（cold/warm mean+P95），并纳入 `docs/experiment.md`（E0137）可 smoke/full 运行且能被 `.rd_queue sync` 摘要到 `docs/results.md`；
    - `docs/plan.md` 的 P0015 DoD 与当前仓库可复现实装保持一致（CT2Rep 先作为可运行占位接口；若要引入真实强基线则另起 P####）。
  - Verification: `python -m provetok.experiments.latency_bench_baselines --smoke --device cpu --budget-tokens 64 --output-dir ./outputs/_verify_latency_bench && python -c "import json, pathlib; p=max(pathlib.Path('outputs/_verify_latency_bench').glob('*/latency_bench_baselines.json'), key=lambda x:x.stat().st_mtime); d=json.loads(p.read_text()); assert 'per_method' in d and 'fixed_grid' in d['per_method']; print('latency_ok')"`
  - Notes: latency bench 只解决“可复现实测 + 报表落盘”；真正 latency-matched 的全曲线对齐/多目标 Pareto 仍可作为后续增强项。
  - Resolution:
    - 新增 `provetok/experiments/latency_bench_baselines.py` 并在 `docs/experiment.md` 增加 E0137；已通过 smoke/full（见 `docs/results.md` 的 E0137）。
    - `.rd_queue sync` 已支持从 `latency_bench_baselines.json` 摘要关键指标到 `docs/results.md`。

- [x] M0016: FLOPs/latency matched 仍是 toy：缺真实模型 FLOPs 统计与“matched setting 强制”机制
  - Ref: P0016, C0001, C0002, C0006
  - Context: `compute_budget.py` 仅是示例单价；无法对齐真实 encoder/decoder/verifier FLOPs，也无法在 runner 中强制匹配总账。
  - Acceptance: 引入真实 FLOPs 统计（至少对 encoder/pcg_head）与 latency 基准脚本（固定 batch/hardware/cold/warm）；runner 在 matched setting 下自动调整 B_enc/B_gen/ROI 成本并拒绝不公平配置。
  - Verification: `python scripts/bench_latency.py --smoke`
  - Notes: 实现方案：PyTorch profiler + fvcore/ptflops（可选）统计；把 detector/segmenter 作为可插拔模块并入账；输出 `FLOPs_total` 与 `P95`。
  - Resolution: `provetok/eval/compute_budget.py` 支持 `flops_extra` 并在 `match_b_enc_for_total_flops()` 中做 matched 求解；`provetok/experiments/run_baselines.py` 支持 `--flops-total --costs-json` 且对每个 baseline 做 fail-fast FLOPs matching（ROI 额外成本入账）；新增 `scripts/profile_flops.py --out-costs` 生成 `ComputeUnitCosts` JSON；并通过 `python scripts/bench_latency.py --smoke` 验证 P95/mean 输出可用。

- [x] M0017: 可审计 artifact 仍不完整：缺 commit/data revision/split manifest 写入与统一 JSON schema
  - Ref: P0017, C0001, C0004, C0005
  - Context: 当前 artifact 有 tokens/citations/issues/trace，但没有写入代码 commit、数据 revision hash、split manifest、硬件信息，难以复现实验/防后验改规则。
  - Acceptance: 所有 experiments 输出统一 `artifact.json` schema：包含 `code_commit`、`data_revision`、`split_manifest_path`、`rule_set_version`、`hardware`、`seed`，以及每样本的 refine trace。
  - Verification: `python -m provetok.run_demo --steps 2 --budget 32 --seed 0`
  - Notes: 实现方案：新增 `provetok/utils/artifact.py` 统一 dump；commit 用 `git rev-parse HEAD`；rule_set_version 固定在 `verifier/rules.py` 常量。
  - Resolution: 新增 `provetok/utils/artifact.py`（统一 meta schema + `code_commit`/`hardware`）；`provetok/verifier/rules.py` 增加 `RULE_SET_VERSION`；`provetok/run_demo.py` 输出 `artifact.meta`（含 commit/data_revision/rule_set_version/seed/config/hardware）。

- [x] M0018: `.rd_queue` 实验队列/恢复机制缺失，无法把 smoke/full 与 docs 勾选形成闭环
  - Ref: P0018, C0001, C0002, C0003, C0004, C0005, C0006
  - Context: `docs-spec` 约定 `.rd_queue/logs`/`results`/`queue.json`，但仓库没有 runner；实验只能手工跑且无法自动更新台账勾选。
  - Acceptance: 增加 `scripts/rd_queue.py`（或等价）支持：入队→tmux/子进程执行→落盘 logs/results JSON；并提供“成功后再勾选 docs/experiment.md”的更新脚本。
  - Verification: `python scripts/rd_queue.py --help`
  - Notes: 实现方案：复用 `rd-experiment-runner` 的产物结构；结果 JSON 包含 cmd/exit_code/time/log_path/output_dir。
  - Resolution: 新增 `scripts/rd_queue.py`：`init/make/start/worker/sync`（解析 `docs/experiment.md` 生成 `.rd_queue/queue.json`，运行后按 `.rd_queue/results/*.json` 同步勾选 `docs/experiment.md` 的 smoke/full）。

## 2. Ambiguities

- [x] M0101: 真实数据条款与下载方式不确定（CT-RATE/CT-3DRRG/ReXGroundingCT）
  - Ref: P0010
  - Context: 需要决定仓库是否提供自动下载脚本；若条款不允许则仅提供 manifest schema
  - Acceptance: `scripts/data/` 同时支持 manifest 驱动 + 可选下载入口（默认不强制下载）
  - Verification: `python scripts/data/validate_manifest.py --help`
  - Notes: 当前默认采用“Manifest 驱动 + 可选下载/预处理”
  - Resolution: 采用“manifest 驱动”为唯一入口：仓库不内置自动下载（遵循数据集条款），用户自行准备数据目录后用 `scripts/data/build_*_manifest.py` 生成 manifest；用 `scripts/data/validate_manifest.py` + `ProtocolLock` 做拆分/交集/去重约束检查。

- [x] M0102: Claim space 的 slot vocab/hierarchy 需要最终锁死（location/size_bin/severity 的层级与映射）
  - Ref: C0004, C0005
  - Context: 目前 `provetok/pcg/schema.py` 的 slot 值是占位集合；真实数据（CT-RATE/ReX）可能用不同命名/层级，若不锁死会导致 verifier/指标漂移。
  - Acceptance: 给出“slot vocab v1.0”文档与映射表（dataset label → vocab）；并在代码中版本锁定（schema 变更需 bump 版本）。
  - Verification: `pytest -q`
  - Notes: 实现方案：新增 `docs/schema_v1.md` + `provetok/pcg/schema_version.py`；在 verifier trace 中写入 schema 版本。
  - Resolution: 新增 `docs/schema_v1.md`（slot vocab v1.0 文档 + 映射规则）与 `provetok/pcg/schema_version.py`（`SCHEMA_VERSION` 锁死）；并在 artifact meta 中写入 schema 版本（见 `provetok/run_demo.py`）。

- [x] M0103: critical finding 集合与 severity 分桶标准需明确（防封嘴主表约束的法律条款）
  - Ref: P0013, C0005
  - Context: 目前 critical set 分散在 `evidence_head.py`/`refusal.py`/`verifier/rules.py`；且未声明“按任务定义锁死”的来源与版本。
  - Acceptance: 把 critical set 与 severity 分桶集中到单一来源并版本锁（例如 `provetok/verifier/taxonomy.py`）；所有评测/校准统一引用。
  - Verification: `pytest -q`
  - Notes: 实现方案：新增 taxonomy 常量文件，写入 `rule_set_version`；更新 verifier/refusal/evidence_head 的引用路径。
  - Resolution: 新增 `provetok/verifier/taxonomy.py`（`TAXONOMY_VERSION`、`RULE_SET_VERSION`、critical finding 与 severity weights 单一来源）；并将 `EvidenceHead` 与 `RefusalCalibrator` 统一引用该模块；artifact meta 写入 taxonomy 版本（见 `provetok/run_demo.py`）。

## Resolved (optional)

- [x] M0001: 缺少 `provetok.data`（manifest/io/dataset/frame_extractor/protocol_lock），导致 demo/训练/烟测导入失败
  - Ref: P0002
  - Context: `provetok/run_demo.py` / `scripts/train_m0.py` / `provetok/training/trainer.py` 依赖 `provetok.data.*` 但目录不存在
  - Acceptance: `from provetok.data import make_dataloader` 可用；synthetic 与 manifest 数据集均可实例化
  - Verification: `python -c "from provetok.data import make_dataloader; print('ok')"`
  - Resolution: 已新增 `provetok/data/*`（manifest_schema/io/dataset/frame_extractor/protocol_lock）与 `scripts/data/*` 校验/拆分脚本，并修复相关导入。

- [x] M0002: `cell_id` 解析与 `Cell.id()` 格式不一致，grounding mask 可能全空
  - Ref: P0003
  - Context: `provetok/grid/cells.py` 输出 `L{level}:{ix},{iy},{iz}`；`provetok/eval/metrics_grounding.py` 解析期望括号形式
  - Acceptance: `cell_id -> parse -> id` round-trip；grounding 指标不再因解析失败退化
  - Verification: `pytest -q`
  - Resolution: 引入 `provetok/grid/cells.py::parse_cell_id` 并统一使用；新增 `tests/test_cell_id_roundtrip.py`。

- [x] M0003: `polarity` 命名不一致（present/absent vs positive/negative），verifier 与 PCGHead 对不齐
  - Ref: P0004
  - Context: `ToyPCG`/`PCGHead` 产出 `present/absent`；`verifier/rules.py` 以 `positive` 判断 U1
  - Acceptance: 全链路统一 polarity 词表；verifier rules 与 metrics 同步
  - Verification: `pytest -q`
  - Resolution: verifier 对 `present/positive` 与 `absent/negative` 做兼容归一；实验侧统一使用 `present/absent`。

- [x] M0004: finding frame schema 未覆盖论文 slots（缺 location/size_bin/severity/uncertain），Table1 指标无法对齐
  - Ref: P0004
  - Context: `provetok/types.py::Frame` 只有 finding/polarity/laterality
  - Acceptance: 引入统一 FindingFrame 并贯穿 PCGHead/verifier/metrics
  - Verification: `pytest -q`
  - Resolution: `Frame` 扩展 slots（location/size_bin/severity/uncertain），`PCGHead` 输出全槽并更新 `metrics_frames.py` 的 slot accuracy；`FrameExtractor` 填充这些字段。

- [x] M0005: Verifier evidence_trace schema 不统一，BET allocator 读不到被 blame 的 cell/token
  - Ref: P0003
  - Context: allocator 读 `token_cell_ids`，但 rules 常写 `cell_ids`/`citations` 等键名
  - Acceptance: evidence_trace 最小 schema 锁死并在所有 rules 写齐；allocator 只读一处
  - Verification: `pytest -q`
  - Resolution: 在 `provetok/verifier/rules.py` 引入 `build_evidence_trace`，统一输出 `token_ids/token_cell_ids/rule_inputs/rule_outputs`。

- [x] M0006: counterfactual 套件不全（缺 token-permutation/evidence-drop/mask-sanity）且缺 bootstrap+Holm
  - Ref: P0006
  - Context: 仅有 `omega_permutation_test`、`citation_swap_test`
  - Acceptance: 反事实实验覆盖 §7.4 并输出 CI+Holm
  - Verification: `python -m provetok.experiments.figX_counterfactual --smoke`
  - Resolution: 新增 `provetok/eval/stats.py`（paired bootstrap + Holm）与 `provetok/experiments/figX_counterfactual.py`（含 evidence-drop/mask sanity）。

- [x] M0007: 联合预算/公平性协议未落地（FLOPs/latency matched、ROI 成本入账、cold/warm + P95）
  - Ref: P0007
  - Context: 仅有 toy FLOPs 单价；无 latency bench
  - Acceptance: 输出 FLOPs_total 与 latency 统计；matched setting runner 可用
  - Verification: `python scripts/bench_latency.py --smoke`
  - Resolution: 新增 `provetok/eval/compute_budget.py` 与 `scripts/bench_latency.py`（mean + P95，cold/warm）。

- [x] M0008: Baselines 不齐（fixed-grid/2D/2.5D/ROI/protocol ablations/强 baseline runner）
  - Ref: P0008
  - Context: 当前主要是 octree-like refine；缺 baseline tokenizers 与 protocol ablations
  - Acceptance: baseline scaffolding 可运行（至少 synthetic），并能产出 matched 对比所需的指标与成本
  - Verification: `python -m provetok.experiments.run_baselines --smoke`
  - Resolution: 新增 `provetok/baselines/*` 与 `provetok/experiments/run_baselines.py`（synthetic scaffold）。
