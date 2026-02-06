# Oral+ Checklist (vNext)

> 本清单面向 “冲顶会 oral” 的下一版补齐：在 `scripts/oral_audit.py --strict` 已显示 `ready_for_oral_gate=true` 的基础上，补齐**最小但决定性**的实验/叙事，以显著降低 reviewer/AC 风险。
>
> P0 的最小集合与现有产物索引：见 `docs/oral_checklist.md`。

## 0) 当前状态（基线）

- 机器裁判（Claims）：`python scripts/proof_check.py --profile default` / `--profile real`
- Oral gate（P0）：`python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict`
- 已通过：C0001–C0006 全部 proved，且 P0（E0160/E0161/E0162/E0163）已全绿（以 `outputs/oral_audit.json` 为准）。

## 1) vNext 最小但决定性 checklist

### [x] V0001: 多目标 Pareto + latency-matched 口径（强化 C0001）

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

### [ ] V0002: “效应大小”与 failure modes（避免只剩 p-value）

- 风险
  - reviewer 会质疑：显著但 effect size 太小，或者指标不具备可解释性；
  - 没有 failure taxonomy/案例会被认为“系统不可控/不可解释”。
- 最小产物（低成本）
  - `outputs/V0002-effect/effect_size_summary.json`（absolute delta、median、分位数、按 subgroup 的 breakdown）
  - `docs/oral_script.md` 增加 1 页 “Failure modes + mitigation” 的口头讲法（引用具体 case 路径）
- 现成可复用证据
  - case triad（正例/拒绝/失败）：`docs/oral_checklist.md` 的 E0163 条目已给出可点开的 `case.json` 路径
  - 证明摘要：`docs/proof_audit.md`
- 路径（如需扩展）
  - 指标与 per-sample 数据通常在 `outputs/**/fig*_raw_data.json`、`outputs/**/baselines.json` 等；如缺少，需在对应 runner 里补齐落盘字段。

### [ ] V0003: 跨数据集 grounding（真正的 generalization）

- 风险
  - 当前 pixel-level grounding 主要在 ReXGroundingCT；CT-RATE 只有 pipeline sanity（缺 `mask_path`），顶会容易被追问泛化。
- 最小可执行路径（按现实约束从易到难）
  - A) 获取一个带 voxel-level mask 的第二数据集，并构建 `manifest.jsonl`（必须含 `mask_path`），再跑 `E0160(smoke)`/`E0162(smoke)`。
  - B) 在 CT-RATE 上人工标注一个小的 eval-only 子集（例如 20–50 volumes 的 lesion masks），只跑评测，不用于训练。
  - C) 使用弱监督 pseudo-mask（例如 `SaliencyCNN3D` 输出阈值化）做“弱标签”评测，但必须在叙事中明确其局限。
- 相关路径
  - manifest schema：`provetok/data/manifest_schema.py`
  - 构建/校验：`scripts/data/build_ct_rate_manifest.py`、`scripts/data/validate_manifest.py`
  - 参考模板：`/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl`（包含 `mask_path`）

### [ ] V0004: 替换 `ct2rep_like` 占位（强化 C0006）

- 风险
  - `ct2rep_like` 是固定 grid 占位，容易被认为“强基线缺失/不公平”。
- 最小解决方案（从轻到重）
  - A) 叙事层面降风险：把 `ct2rep_like` 明确定位为 “no-citations ablation placeholder”，并在 `docs/plan.md`/`README.md` 里强调不等价于真实 CT2Rep。
  - B) 工程层面降风险：重命名 `ct2rep_like` 为 `ct2rep_placeholder`（同步更新 `scripts/proof_check.py`/runner/文档），避免误导。
  - C) 论文层面彻底解决：实现并训练一个公开可复现的强 3D RRG baseline（或引入公开实现与权重），并纳入 `run_baselines` + `proof_check` 的 required list。
- 相关路径
  - baseline suite：`provetok/experiments/run_baselines.py`
  - 训练强 baseline：`provetok/experiments/train_ct2rep_strong.py`
  - 判定规则：`scripts/proof_check.py` + `docs/plan.md`（C0006）

### [ ] V0005: Public reproducibility（GitHub 交付）

- 风险
  - 顶会 oral 常被追问 “能不能一键复现/能不能审计”；repo 不 clean 会直接加大信任成本。
- 最小产物
  - `README.md` 的 “reproduce oral P0” 一键路径（含 `.rd_queue`、`proof_check`、`oral_audit`）
  - 清理无关文件（例如 `node_modules/` 不应入库），并通过 SSH push 到 GitHub
- 相关路径
  - 可复现入口：`README.md`、`docs/experiment.md`、`scripts/rd_queue.py`
  - 审计入口：`scripts/oral_audit.py`

## 2) “完整的风险→解决方案”目录（不重复贴）

- 每条 Claim 的 “为什么不 work + 可尝试修复方向（尽量全）”：`docs/proof_strength_report.md`
- 更细的失败复盘与已做修复记录：`docs/proof_failure_analysis.md`
