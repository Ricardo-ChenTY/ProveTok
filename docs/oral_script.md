# Oral Script (Top-Conference Ready) — ProveTok

> 目标：给你一份**可直接照念/照讲**的 oral 讲稿骨架（主线 + 反质疑 + 复现/审计指引），并把每句话都对应到可跑的 artifacts。

## 0) Two-sentence contributions (B1)

1) **BET（预算显式化 + allocation）**：我们把 3D→text 的证据提取写成显式联合预算 `B=B_enc+B_gen` 的优化问题，并在 FLOPs/latency-matched 协议下给出可解释的 scaling 与跨预算配置选择（regret+CI）。
2) **PCG（proof-carrying + refusal anti-gaming）**：我们把 groundedness 从“事后评测”改成**协议**：每条断言必须携带可机检 citations 与可审计 verifier trace；证据不足时触发可约束 refusal，并用 critical miss-rate gate 防止“封嘴”。

## 1) Main-figure walkthrough in 6 sentences (B2)

1) 先看 **Fig2 scaling/Pareto**：预算越大并不等价于质量越好，我们报告多预算曲线并把 tail latency（warm P95）纳入硬 gate。  
2) 我们不是只看一个点：在多个 budgets 上用 paired bootstrap + Holm 判定是否真实 dominate（见 `python scripts/proof_check.py`）。  
3) 接着看 **grounding**：每条 finding 的 citations 映射到 3D 区域 Ω，和像素级 mask 做 IoU/hit-rate；我们要求对多个 baselines 都显著提升。  
4) 然后看 **counterfactual**：swap/drop/permutation 反事实会系统性击穿（unsupported↑ / IoU↓ / correctness↓），证明 citations 不是装饰。  
5) 最后是 **refusal**：当证据不足时系统拒答，但 critical miss-rate 不上升，且校准可控（ECE/可靠性）。  
6) 全流程 artifacts 可审计：每个实验输出包含 `meta`（commit/data_revision/rule_set/schema/taxonomy），可一键重跑并逐 sample 复盘。

## 1.5) Fig3 allocation：避免“后验挑配置”的口径（E0161）

口头问：“你这不是后验挑一个最好的配置/曲线吗？”  
回答（固定口径）：

1) **候选空间预先锁定**：Fig3 的 tags（`b_gen/n_verify/topk`）在跑 test 之前就固定写进实验脚本与台账（见 `docs/oral_checklist.md` 的 E0161）。  
2) **dev/test 严格分离**：只用 `split=val` 的 dev curves 拟合/选择 allocation policy；`split=test` 只做一次性评估，任何阈值/超参不在 test 上调。  
3) **报告 regret+CI（不是挑点）**：用 `outputs/E0161-full/fig3_regret_sweep.json` 的 mean regret + bootstrap CI 证明“选配策略”整体优于 naive policy，而不是挑一个预算点讲故事。

## 2) 4 anti-reviewer questions (B3)

1) **“是不是封嘴赢指标？”**  
   - 回答：refusal 有硬 gate：`critical miss-rate<=δ`、`refusal_rate<=0.20`、`refusal_ece<=0.15`；并报告阈值如何在 dev 校准一次后跨 budgets 固定（见 `outputs/E0144-full/figX_refusal_calibration.json`）。

2) **“verifier 太弱/太随意？”**  
   - 回答：verifier 是显式规则集（versioned），输出 issue types + evidence trace；counterfactual 中用 relevance-only 的 U1.4 作为 primary unsupported 口径，避免混入 coverage/uncertainty confounders（见 `provetok/experiments/figX_counterfactual.py` 的 `unsupported_rates` vs `unsupported_rates_full`）。

3) **“只看 mean latency 没意义，tail 爆炸怎么办？”**  
   - 回答：我们在 proof rule 中把 warm P95 当 hard gate；任何预算点 tail 超过 baseline +5% 则判失败（见 `docs/plan.md` 的 C0001 rule + `python scripts/proof_check.py` 输出）。

4) **“只是 toy/scaffold，真实 end-to-end 不 work？”**  
   - 回答：oral-ready P0 要求补齐真实数据上的 end-to-end（LLM+3D encoder）曲线与反事实（见 `docs/oral_checklist.md` 的 E0160/E0162/E0163），并提供可审计 case study 作为现场打开解释的证据。

## 3) Repro / Audit one-pager (B4)

### 3.1 Install + tests

```bash
pip install -r requirements.txt
pytest -q
python scripts/proof_check.py
```

### 3.2 Run via `.rd_queue` (recommended)

```bash
python scripts/rd_queue.py make --stage smoke --out .rd_queue/queue_smoke.json
python scripts/rd_queue.py start --queue .rd_queue/queue_smoke.json --session rdq_smoke
python scripts/rd_queue.py sync
```

### 3.3 Oral-ready P0 runs (current checklist)

- **E0160** (real Fig2): `outputs/E0160-*/fig2_multiseed.json`  
- **E0161** (non-trivial allocation): `outputs/E0161-*/baselines_curve_multiseed.json` + `outputs/E0161-*/fig3_regret_sweep.json`  
- **E0162** (counterfactual): `outputs/E0162-*/figX_counterfactual.json`  
- **E0163** (case studies): `outputs/E0163-*/case_<scan_hash>/{case.json,case.png}`  

建议使用：

```bash
python scripts/rd_queue.py make --stage full --ids E0162 --out .rd_queue/queue_E0162_full.json
python scripts/rd_queue.py start --queue .rd_queue/queue_E0162_full.json --session rdq_E0162_full
python scripts/rd_queue.py sync
```

### 3.4 What to open during an oral Q&A

1) `docs/results.md`：每条实验的结束时间、状态、输出 artifact 路径与关键指标摘要  
2) `outputs/E0163-full/index.json`：case 索引；打开某个 `case.json` 看 tokens/citations/issues/trace；配合 `case.png` 解释 Ω 与 lesion 的空间关系  
3) `python scripts/proof_check.py`：机器裁判输出（防止“口说无凭”）
