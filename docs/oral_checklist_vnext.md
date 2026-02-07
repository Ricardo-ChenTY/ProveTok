# Oral+ Checklist (vNext)

> 本清单面向 “冲顶会 oral” 的下一版补齐：截至 2026-02-07 05:27 UTC，`scripts/oral_audit.py --strict` 已显示 `ready_for_oral_gate=true`（`gaps=[]`，含 `real::C0001`）。
>
> 当前目标从“堵 C0001 缺口”切换为“稳固口径 + 扩展泛化（V0003）”；P0 的最小集合与现有产物索引见 `docs/oral_checklist.md`。

## 0) 当前状态（基线）

- 机器裁判（Claims）：`python scripts/proof_check.py --profile default` / `--profile real`
- Oral gate（P0）：`python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict`
- 当前结论：`default` 与 `real` 档均为 C0001–C0006 全部 proved；oral gate 已通过。
- 关键数值（`real::C0001`）：`combined_pass=6/6`、`iou_pass=6/6`、`latency_p95_pass=6/6`、`unsupported_pass=6/6`（见 `python scripts/proof_check.py --profile real`）。
- 口径锚点：`outputs/oral_audit.json`（更新时间：2026-02-07 05:27 UTC，`ready_for_oral_gate=true`）。

## 1) vNext 最小但决定性 checklist

### [x] V0001: 多目标 Pareto + latency-matched 口径（修复 C0001(real)）

- 风险（顶会常见质疑）
  - “Pareto dominate” 可能被认为只是 hard gate + 单指标显著，而不是真正多目标 Pareto frontier；
  - latency-matched 口径容易被质疑不一致（cold/warm、mean/P95、是否包含 I/O）。
- 最小产物（无需新训练）
  - `outputs/V0001-pareto/pareto_report.json`（可审计表格化：每 budget×method 的 quality/trust/latency）
  - `outputs/V0001-pareto/pareto_table.md`（可直接贴到 rebuttal/appendix 的表）
  - 当前已生成：
    - `outputs/V0001-pareto/pareto_report-default_profile.json` + `outputs/V0001-pareto/pareto_table-default_profile.md`
    - `outputs/V0001-pareto/pareto_report-real_profile.json` + `outputs/V0001-pareto/pareto_table-real_profile.md`
- 代码路径（新增）
  - `scripts/pareto_report.py`（从已有 `baselines_curve_multiseed.json` 生成 multi-objective report）
- 运行路径（示例）
  - `python scripts/pareto_report.py --in outputs/E0138-full/baselines_curve_multiseed.json --out outputs/V0001-pareto`
  - `python scripts/pareto_report.py --in outputs/E0164-full/baselines_curve_multiseed.json --out outputs/V0001-pareto --tag real_profile`
- Done when（最小通过条件）
  - 明确写出：objective 列表、dominance 判定、以及每个 budget 的 Pareto frontier（含 trust metrics 与 warm P95）。
  - `python scripts/proof_check.py --profile real` 中 C0001 由 not proved 变为 proved（`combined/iou >=4/6` 且 hard gate 全通过）。
  - 当前结果：`real::C0001` 已为 proved（`combined/iou/latency_p95/unsupported` 全为 `6/6` 通过）。

### [x] V0001R: C0001(real) 关闭路径（执行版）

- 路径 A（统计功效）
  - 在不改判定规则前提下，把 E0164 的 `n_samples` 从 100 提升到 150/200，保持 `6 budgets x 5 seeds x bootstrap=20000`，重跑并对比 `combined/iou` 的 pass budget 数。
- 路径 B（latency tail）
  - 专门定位历史上 `B=3e6` 的 `warm_p95` 超标（曾约 +8.4%），优先优化 `provetok_lesionness` 的 score/citation 计算路径，目标恢复 `<=+5%`。
- 路径 C（效应量）
  - 在固定真实基线 `ct2rep_noproof` 下做预注册小网格：`citation_strategy`、`score_fuse`、`topk_citations`，只接受事前写入 `docs/experiment.md` 的组合，避免后验挑点。
- 路径 D（停机条件）
  - `outputs/oral_audit.json` 的 `gaps` 为空且 `ready_for_oral_gate=true`，并在 `docs/results.md` 出现对应 rerun 条目。
- 执行结果（2026-02-06）
  - 关闭条件已满足：`python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict` 返回 0，`gaps=[]`，`ready_for_oral_gate=true`。
  - 证明口径与 artifact 路径已锁定：`outputs/E0164-full/baselines_curve_multiseed.json`。

### [x] V0002: “效应大小”与 failure modes（避免只剩 p-value）

- 风险
  - reviewer 会质疑：显著但 effect size 太小，或者指标不具备可解释性；
  - 没有 failure taxonomy/案例会被认为“系统不可控/不可解释”。
- 最小产物（低成本）
  - `outputs/V0002-effect/effect_size_report.json`（从 `scripts/proof_check.py` 提取 per-budget delta/CI/门槛通过情况，避免只剩 p-value）
  - `outputs/V0002-effect/effect_size_table.md`（可直接贴到 appendix/rebuttal）
  - `docs/oral_script.md` 增加 1 页 “Failure modes + mitigation” 的口头讲法（引用具体 case 路径）
- 当前已生成（见上方两个文件）
- 现成可复用证据
  - case triad（正例/拒绝/失败）：`docs/oral_checklist.md` 的 E0163 条目已给出可点开的 `case.json` 路径
  - 证明摘要：`docs/proof_audit.md`
- 路径（如需扩展）
  - 指标与 per-sample 数据通常在 `outputs/**/fig*_raw_data.json`、`outputs/**/baselines.json` 等；如缺少，需在对应 runner 里补齐落盘字段。

### [x] V0003: 跨数据集 grounding（真正的 generalization，先完成弱标签 C 路径）

- 风险
  - 当前 pixel-level grounding 主要在 ReXGroundingCT；CT-RATE 只有 pipeline sanity（缺 `mask_path`），顶会容易被追问泛化。
- 最小可执行路径（按现实约束从易到难）
  - A) 获取一个带 voxel-level mask 的第二数据集，并构建 `manifest.jsonl`（必须含 `mask_path`），再跑 `E0160(smoke)`/`E0162(smoke)`。
  - B) 在 CT-RATE 上人工标注一个小的 eval-only 子集（例如 20–50 volumes 的 lesion masks），只跑评测，不用于训练。
  - C) 使用弱监督 pseudo-mask（例如 `SaliencyCNN3D` 输出阈值化）做“弱标签”评测，但必须在叙事中明确其局限。
- 已执行（2026-02-06，C 路径）
  - 新增脚本：`scripts/data/build_ct_rate_pseudomask_manifest.py`
  - 生成弱标签 manifest（CT-RATE test=30）：
    - `python scripts/data/build_ct_rate_pseudomask_manifest.py --in-manifest /data/provetok_datasets/ct_rate_100g/manifest.jsonl --out-manifest /data/provetok_datasets/ct_rate_100g_pseudomask/manifest.jsonl --pseudo-mask-dir /data/provetok_datasets/ct_rate_100g_pseudomask/masks --saliency-weights outputs/E0155-train_saliency_cnn3d_100g/saliency_cnn3d.pt --splits test --max-samples-per-split 30 --resize-shape 64 64 64 --device cpu --overwrite`
  - 校验通过：`python scripts/data/validate_manifest.py --manifest /data/provetok_datasets/ct_rate_100g_pseudomask/manifest.jsonl`
  - 评测 smoke（grounding）：`outputs/E0165-ct_rate-pseudomask-smoke/figX_grounding_proof.json`
    - `provetok_lesionness` vs `roi_variance`：`iou_union mean_diff=+0.0332`, `p_holm=0.008`（通过）
    - `provetok_lesionness` vs `fixed_grid`：`iou_union mean_diff=+0.0255`, `p_holm=0.128`（smoke 样本下未过 Holm）
  - 评测 smoke（counterfactual）：`outputs/E0165-ct_rate-pseudomask-counterfactual-smoke/figX_counterfactual_20260206_195346/figX_counterfactual.json`
  - 评测 preflight（grounding，多预算）：
    - `outputs/E0165-ct_rate-pseudomask-preflight/figX_grounding_proof.json`
    - budgets=`{2e6,5e6,7e6}`、n_samples=30、seed=0、bootstrap=2000 下，`iou_union` 对 `fixed_grid/roi_variance` 均为正且 Holm 后显著（3/3 budgets）。
  - 评测 preflight（counterfactual）：
    - `outputs/E0165-ct_rate-pseudomask-counterfactual-preflight/figX_counterfactual_20260206_195836/figX_counterfactual.json`
    - `omega_perm` 与 `no_cite` 在 `grounding_iou_union_orig_minus_cf` 上显著（Holm 后 `p=0.0`）；`unsupported` 相关项在该弱标签设定下未出现显著变化（需在口头中明确）。
  - 评测 full（grounding，58 test 全量 + 多 seed）：
    - 运行记录：`.rd_queue/results/E0165-full.json`（`status=passed`，`started_at=2026-02-07T04:05:12+00:00`，`ended_at=2026-02-07T04:10:06+00:00`）
    - 主结果：`outputs/E0165-ct_rate-pseudomask-full/figX_grounding_proof.json`
    - `iou_union` 配对 bootstrap + Holm（6 budgets × 3 seeds）：
      - vs `roi_variance`：`mean_diff>0` 且 Holm 显著为 `6/6` budgets
      - vs `fixed_grid`：`mean_diff>0` 为 `6/6` budgets，Holm 显著为 `5/6` budgets（`3e6` 预算 `p_holm=0.5212`）
  - 评测 full（counterfactual，58 test 全量）：
    - 运行记录：`.rd_queue/results/E0165CF-full.json`（`status=passed`，`started_at=2026-02-07T04:44:23+00:00`，`ended_at=2026-02-07T04:45:52+00:00`）
    - 主结果：`outputs/E0165-ct_rate-pseudomask-counterfactual-full/figX_counterfactual_20260207_044425/figX_counterfactual.json`
    - `grounding_iou_union_orig_minus_cf`：
      - `omega_perm`: `mean_diff=+0.005398`, `p_holm=0.0308`（显著）
      - `no_cite`: `mean_diff=+0.009464`, `p_holm=0.0`（显著）
      - `cite_swap`: `mean_diff=0.0`, `p_holm=1.0`（不显著）
  - 评测 full（counterfactual，多 seed 稳定性）：
    - 运行记录：`.rd_queue/results/E0165CF1-full.json`、`.rd_queue/results/E0165CF2-full.json`（均 `status=passed`）
    - 产物：`outputs/E0165-ct_rate-pseudomask-counterfactual-full-seed1/figX_counterfactual_20260207_045329/figX_counterfactual.json`、`outputs/E0165-ct_rate-pseudomask-counterfactual-full-seed2/figX_counterfactual_20260207_045457/figX_counterfactual.json`
    - `grounding_iou_union_orig_minus_cf`：
      - `omega_perm` 在 seed `{0,1,2}` 的 `mean_diff={+0.005398,+0.005878,+0.006135}`，Holm 后 `p={0.0308,0.0048,0.0024}`（`3/3` 显著）
      - `no_cite` 在 seed `{0,1,2}` 的 `mean_diff` 固定为 `+0.009464`，Holm 后 `p=0.0`（`3/3` 显著）
      - `cite_swap` 在 seed `{0,1,2}` 均为 `mean_diff=0.0`、`p=1.0`
    - `unsupported`/`overclaim` 在该弱标签设定下保持退化状态（分别接近全 1 / 全 0），不构成可分辨信号。
  - 已执行（2026-02-07，A' 路径：外部 TS-Seg auto mask，eval-only）：
    - 构建脚本：`scripts/data/build_ct_rate_tsseg_effusion_manifest.py`；生成 `manifest=/data/provetok_datasets/ct_rate_tsseg_effusion_eval/manifest.jsonl`（`n=38`，`split=test`，`mask_quality=silver_auto_unverified`）。
    - 运行记录（grounding）：`.rd_queue/results/E0166-full.json`（`status=passed`，`started_at=2026-02-07T05:44:37+00:00`，`ended_at=2026-02-07T05:48:01+00:00`）。
    - 主结果（grounding）：`outputs/E0166-ct_rate-tsseg-effusion-grounding-full/figX_grounding_proof.json`
      - `iou_union` 对 `roi_variance`：`mean_diff>0` 且 Holm 显著 `6/6` budgets；
      - `iou_union` 对 `fixed_grid`：`mean_diff>0` 为 `5/6` budgets、Holm 显著 `4/6` budgets（`4e6` 为负、`5e6` 未过 Holm）。
    - 运行记录（counterfactual 多 seed）：`.rd_queue/results/E0167S0-full.json`、`.rd_queue/results/E0167S1-full.json`、`.rd_queue/results/E0167S2-full.json`（均 `status=passed`）。
    - 主结果（counterfactual）：`outputs/E0167-ct_rate-tsseg-effusion-counterfactual-full-seed{0,1,2}/figX_counterfactual_*/figX_counterfactual.json`
      - `no_cite` 在 seed `{0,1,2}` 均为 `mean_diff=+0.005909`，Holm 后 `p=0.0`（`3/3` 显著）；
      - `omega_perm` 在 seed `{0,1,2}` 为 `mean_diff={+0.001515,+0.002341,+0.003146}`，Holm 后 `p={1.0,1.0,0.5128}`（方向一致但未显著）；
      - `cite_swap` 在 seed `{0,1,2}` 均为 `mean_diff=0.0`、`p=1.0`；`unsupported/overclaim` 差值均为 0（无可分辨信号）。
  - 已执行（2026-02-07，A' omega_perm 功效增强）：
    - 运行记录（扩 seed）：`.rd_queue/results/E0167S3-full.json` ~ `.rd_queue/results/E0167S9-full.json`（均 `status=passed`；总 seeds=`{0..9}`）。
    - 汇总脚本：`scripts/analysis/counterfactual_omega_perm_power.py`
    - 汇总产物：`outputs/E0167R-ct_rate-tsseg-effusion-counterfactual-power/omega_perm_power_report.json` 与 `omega_perm_power_table.md`
    - pooled primary（预注册 one-sided）：
      - `grounding_iou_union_orig_minus_cf::omega_perm`: `mean_diff=+0.002007`, `95% CI=[+0.000130,+0.003680]`, `p_one_sided=0.0187`（通过）
      - 方向一致性：`positive_seeds=9/10`
    - pooled secondary（family-wise Holm across cf keys）：
      - `omega_perm p_holm=0.1122`（未过），`no_cite p_holm=0.0`（通过）
- 局限（必须口头声明）
  - 该路径是弱标签评测，不等价于 gold voxel mask；只用于“跨集可运行 + 趋势参考”，不能替代主结论。
  - 当前 CT-RATE 域偏移明显，`threshold=0.5` 退化后频繁触发 top-k fallback（mean mask ratio≈0.005）；说明 pseudo-mask 质量受限。
  - TS-Seg 路径同样不是 gold mask，且来源为自动分割（`silver_auto_unverified`）；它提升了“外部来源独立性”，但仍不能替代人工/官方标注证据。
  - 要把 V0003 升级为强证据，仍建议推进 A/B（真实 mask 或人工标注 eval 子集）。
- 相关路径
  - manifest schema：`provetok/data/manifest_schema.py`
  - 构建/校验：`scripts/data/build_ct_rate_manifest.py`、`scripts/data/validate_manifest.py`
  - 参考模板：`/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl`（包含 `mask_path`）

### [x] V0004: 替换 `ct2rep_like` 占位（强化 C0006）

- 风险
  - 占位基线若只是 fixed-grid 包装，容易被误解为“真实 CT2Rep”并触发公平性质疑。
- 最小解决方案（从轻到重）
  - A) 叙事层面降风险：把 “无 proof-carrying 对照” 与 “真实强基线” 分开命名，避免误导。
  - B) 工程层面降风险：把占位 fixed-grid 去掉，改成真实模型基线 `ct2rep_noproof`（与 `ct2rep_strong` 共用训练权重，但禁用 citation/refusal）。
  - C) 论文层面彻底解决：实现并训练一个公开可复现的强 3D RRG baseline（或引入公开实现与权重），并纳入 `run_baselines` + `proof_check` 的 required list。
- 当前状态
  - 已完成 A+B：`run_baselines` 中已移除 fixed-grid 占位，改为真实模型基线 `ct2rep_noproof`；`latency_bench_baselines` 中也移除占位方法，避免误导。
  - 仍可继续做 C（真正公开强基线）以进一步提升说服力。
- 相关路径
  - baseline suite：`provetok/experiments/run_baselines.py`
  - 训练强 baseline：`provetok/experiments/train_ct2rep_strong.py`
  - 判定规则：`scripts/proof_check.py` + `docs/plan.md`（C0006）

### [x] V0005: Public reproducibility（GitHub 交付）

- 风险
  - 顶会 oral 常被追问 “能不能一键复现/能不能审计”；repo 不 clean 会直接加大信任成本。
- 最小产物
  - `README.md` 的 “reproduce oral P0” 一键路径（含 `.rd_queue`、`proof_check`、`oral_audit`）
  - 清理无关文件（例如 `node_modules/` 不应入库），并通过 SSH push 到 GitHub
- 当前状态
  - 已 push 到 `origin/main`（commit: `7df59bf`）。
- 相关路径
  - 可复现入口：`README.md`、`docs/experiment.md`、`scripts/rd_queue.py`
  - 审计入口：`scripts/oral_audit.py`

## 2) “完整的风险→解决方案”目录（不重复贴）

- 每条 Claim 的 “为什么不 work + 可尝试修复方向（尽量全）”：`docs/proof_strength_report.md`
- 更细的失败复盘与已做修复记录：`docs/proof_failure_analysis.md`
