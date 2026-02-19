# Gap Report (2026-02-06)

本文件用于回答：“根据当前项目，完整列出还没做完的事情（实现差异 + 未跑实验）并落到 `docs/plan.md` / `docs/mohu.md` / `docs/experiment.md` 里”。

## 0) Snapshot

- Repo: `/home/ubuntu/tiasha/ProveTok`
- Git HEAD: `git rev-parse --short HEAD`（以当前工作区为准）
- Tests: `pytest -q` ✅（15 passed）
- Proof: `python scripts/proof_check.py` ✅（C0001–C0006 全部 proved，按最小 proof rule）
- GPU: `NVIDIA RTX A6000` ✅
- Data roots:
  - `/data/provetok_datasets/rexgroundingct_100g`
  - `/data/provetok_datasets/rexgroundingct_mini`
  - `/data/provetok_datasets/ct_rate_100g`
  - `/data/models/Llama-2-7b-chat-hf`
- `docs/results.md`：已同步 `.rd_queue/results/*.json` 的结果摘要（以 `python scripts/rd_queue.py sync` 为准）。
- Oral P0（状态）：
   - ✅ `E0161 (full)` passed：`outputs/E0161-full/fig3_regret_sweep.json`
   - ✅ `E0162 (full)` passed：`outputs/E0162-full/**/figX_counterfactual.json`
   - ✅ `E0163 (full)` passed：`outputs/E0163-full/index.json`
   - ✅ `E0160P (full)` passed（preflight）：`outputs/E0160-preflight/fig2_multiseed.json`
   - ✅ `E0160 (full)` passed：`outputs/E0160-full/fig2_multiseed.json`

## 1) Repo 现状（“整个项目”）

### 1.1 目录结构（关键模块）

- `provetok/data/*`：manifest 驱动数据集、ProtocolLock、frame extractor、I/O
- `provetok/bet/*`：BET refine loop（含 cache-aware TokenEncoder）、EvidenceHead、allocator
- `provetok/pcg/*`：ToyPCG + LLaMA2 PCG（多帧 JSON 抽取/清洗），schema/version 锁
- `provetok/verifier/*`：规则集、taxonomy、NLI（占位/可插拔）
- `provetok/experiments/*`：Fig2 scaling / Fig3 allocation / Counterfactual / Refusal / Baselines runner
- `scripts/*`：`.rd_queue` 运行工具、FLOPs profiler、latency bench、数据集 manifest 构建、训练入口

### 1.2 当前闭环状态（最小证明已闭环）

- 证据链：`docs/plan.md`（Claims）→ `docs/experiment.md`（E####）→ `outputs/*`（artifacts）→ `docs/results.md`（摘要）→ `python scripts/proof_check.py`（自动判定）。
- 失败分析与备选修复方案：见 `docs/proof_failure_analysis.md`（含每条 Claim 的“为什么不 work + 解决方案清单 + 已做的修复记录”）。

## 2) 仍未达到的“更强版本/论文级”差异（按 Claim 对齐）

> 说明：以下均为“更强版本”的 gap；不影响当前 `docs/plan.md` 的最小 proof rule 已通过。

### 2.1 C0001（Pareto dominate in FLOPs/latency-matched）

更强版本缺口：
- **完整 latency-matched**：
  - ✅ baseline 侧同协议 latency 报表已系统化（cold/warm mean + P95）：E0137（见 `docs/results.md`）。
  - ⏳ 仍缺：把 latency-matched 真正纳入多预算对齐（而不只是报告/附录），以及 baseline 侧与 ProveTok 侧的“同协议、同口径”的 matched 判定。
- **多目标 Pareto**：把 trust metrics（unsupported/overclaim/refusal/ECE + latency）纳入 Pareto frontier，并给出 dominance 判定。

### 2.2 C0003（counterfactual non-triviality）

更强版本缺口：
- ✅ **optional stronger check 已补齐：non-oracle `omega_perm` 现在可在 `iou_union` 与 `iou_max` 两个口径上显著击穿 grounding**（见 `docs/proof_audit.md` 的 C0003 “Optional stronger check”，以及对应的 `outputs/E0157-full/**/figX_counterfactual.json`）。
- 仍可进一步增强（更像“顶会 oral”而非“能过最低线”）：
  - 做 multi-seed / 更大样本复验（目前 stronger check 依赖更强 citations↔lesion 对齐策略，仍可能被质疑为“个例”）。
  - 把该 stronger check 升格为主 proof rule（需要同步更新 `docs/plan.md` + `scripts/proof_check.py`，避免后验）。

### 2.3 C0006（baseline completeness + fairness）

更强版本缺口：
- **强 3D RRG baseline**：`ct2rep_like` 仍是协议占位，建议替换为公开可复现强基线。
- **跨域闭环**：CT-3DRRG 的端到端训练/评测与 matched 对齐仍建议补齐（作为 “generalization” 结果）。

## 4) vNext（Oral+）最小决定性补齐清单

如果目标从 “P0 oral-ready” 升级为 “冲顶会 oral（更强版本）”，建议按 `docs/oral_checklist_vnext.md` 的 checklist 逐条补齐（含每条 risk 的可执行路径/产物/通过条件）。

## 3) 工程闭环的小修复（.rd_queue）

- `scripts/rd_queue.py make` 已支持 **按 stage 选择命令**：smoke 优先 `1GPU script`，full 优先 `Multi-GPU script`（兼容“smoke/full 命令不同”的 E0122–E0129）。
- `docs/results.md` 摘要提取已覆盖 `train_lesionness_head.json` / `figX_grounding_proof.json` / `fig3_results.json`。
