# ProveTok

ProveTok: **Proof-Carrying Budgeted Evidence Tokenization (BET) + Proof-Carrying Generation (PCG)** for grounded 3D CT report generation.

核心闭环（可中断/可恢复/可审计）：
**Tokenize(BET) → Generate(PCG: frames+citations) → Verify → Refine**，并将状态落在 `docs/*.md` + `.rd_queue/`。

---

## 0) 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

快速自检（推荐每次改动后都跑）：

```bash
pytest -q
python scripts/proof_check.py
```

---

## 1) Demo（合成数据，跑通可审计 artifact）

```bash
python -m provetok.run_demo --steps 3 --budget 64 --seed 0
```

输出包含 tokens/Ω、frames+citations、verifier issues、refine trace、版本锁信息等。

---

## 2) 数据（Manifest 驱动）

本仓库采用“manifest 为唯一入口”的数据策略：你准备好本地数据目录后，先构建 `manifest.jsonl` 再跑训练/实验。

常用脚本：

```bash
python scripts/data/validate_manifest.py --help
python scripts/data/build_ct_rate_manifest.py --help
python scripts/data/build_rex_groundingct_manifest.py --help
python scripts/data/build_ct_3drrg_manifest.py --help
```

---

## 3) 实验（`docs/experiment.md` + `.rd_queue`）

实验台账在 `docs/experiment.md`，队列运行/同步用：

```bash
python scripts/rd_queue.py make --stage smoke --out .rd_queue/queue.json
python scripts/rd_queue.py start --queue .rd_queue/queue.json --session rdq_smoke
python scripts/rd_queue.py sync
```

也可以直接手工跑单个实验（示例）：

```bash
# Fig2 scaling
python -m provetok.experiments.fig2_scaling_law --dataset-type manifest --manifest /path/to/manifest.jsonl --split test --budgets 16 32 64 --n-samples 10 --no-plot

# Baselines + matched accounting (toy unit-cost model)
python scripts/profile_flops.py --device cpu --out ./outputs/flops_profile.json --out-costs ./outputs/compute_costs.json
python -m provetok.experiments.run_baselines --flops-total 1000 --costs-json ./outputs/compute_costs.json --dataset-type synthetic --n-samples 5
```

常用开关：
- 数据：`--dataset-type synthetic|manifest` + `--manifest ...`
- PCG：`--pcg toy|llama2`（LLM 需 `--llama2-path ...`）
- Encoder：`--encoder toy|cnn3d`（示例 encoder）

### Oral-ready（P0 最小集合）

Oral checklist 与讲稿在：
- `docs/oral_checklist.md`
- `docs/oral_script.md`

推荐用队列跑（会落日志到 `.rd_queue/logs/`、结果到 `.rd_queue/results/`，并可 `sync` 回写台账）：

```bash
# 例：跑 E0162 full
python scripts/rd_queue.py make --stage full --ids E0162 --out .rd_queue/queue_E0162_full.json
python scripts/rd_queue.py start --queue .rd_queue/queue_E0162_full.json --session rdq_E0162_full
python scripts/rd_queue.py sync
```

> 说明：`outputs/` 与 `.rd_queue/` 默认不入库（见 `.gitignore`）。可复现路径以 `docs/experiment.md`（命令）与 `docs/results.md`（产物路径）为准。

---

## 4) 训练（统一 Trainer 入口）

`scripts/train_m0.py` 已改为统一 Trainer 的训练入口（stage 可切换，支持 BET refine loop 的 stage 参数）：

```bash
python scripts/train_m0.py --config configs/m0.yaml --stage M1 --device cpu
```

---

## 5) 代码导航

- 数据：`provetok/data/*`（manifest schema / protocol lock / dataset）
- BET：`provetok/bet/*`（tokenize / refine_loop / evidence_head）
- PCG：`provetok/pcg/*`（toy + llama2 backend / narrative round-trip）
- Verifier：`provetok/verifier/*`（rules + taxonomy）
- Experiments：`provetok/experiments/*`

## 6) 文档索引

- `docs/plan.md`：Claims（C0001–C0006）+ proof rules（paper-grade）
- `docs/experiment.md`：实验台账（E####）+ 可运行命令（smoke/full）
- `docs/results.md`：从 `.rd_queue/results/*.json` 同步的结果摘要（产物路径 + 指标摘要）
- `docs/proof_audit.md` / `docs/proof_strength_report.md`：proof 判定逻辑与强弱评估
