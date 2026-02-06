# Oral-Ready Checklist (NeurIPS/ICML)

> 目标：把当前“最小证明已闭环”的系统，补齐到 **顶会 oral 可讲、可复现、可抗质疑** 的最小但决定性实验/叙事清单。
>
> 原则：只做能显著降低 reviewer/area-chair 风险的“硬补齐”，避免堆无关实验。

## 0) 口径（什么叫 “oral-ready done”）

- **D0（必须）**：`E0160(full)+E0161(full)+E0162(full)` 都跑通，且每个实验都能在 1 张图/表里回答一个“硬质疑”。
- **D1（必须）**：至少 3 个**可审计 qualitative case**（含 tokens/Ω、citations、verifier trace、refusal），能在 oral Q&A 中“当场点开解释”。
- **D2（必须）**：`docs/results.md` + `outputs/**/meta` 完整版本锁（commit/data_revision/split_manifest/rule_set_version），一键重跑路径明确。

## A) 决定性实验（最小集合）

### A1. Real end-to-end（主图 Fig2 / Pareto / scaling）
- [ ] **E0160 (NEW)**：`pcg=llama2` + `encoder=cnn3d` 在 `ReXGroundingCT-100g` 的多预算曲线（FLOPs-matched + latency 报表）
  - 目的：把“toy→LLM”补齐到真实生成器，避免被质疑为 scaffolding。
  - Done when（最小）：
    - [x] Smoke 通过：输出 `outputs/E0160-smoke/fig2_multiseed.json`
    - [x] Full 通过：输出 `outputs/E0160-full/fig2_multiseed.json`
    - [x] 报告中包含：`warm_p95_s`（tail latency）+ `flops_total`（matched 口径）+ `per_seed` 路径
  - 要求（最小口径）：
    - budgets：`{2e6, 5e6, 7e6}`（可扩到 6 budgets 做 paper-grade）
    - seeds：`{0,1,2}`（≥3）
    - 输出：`fig2_multiseed.json`（含 CI）+ `fig2_raw_data.json`（含 per-sample）
    - 指标：Frame-F1 / IoU_union / unsupported / warm P95 latency
  - 运行（smoke）：
    - `python -m provetok.experiments.fig2_scaling_multiseed --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --split test --resize-shape 64 64 64 --budget-mode flops --budgets 2000000 5000000 7000000 --costs-json outputs/compute_costs.json --pcg llama2 --llama2-path /data/models/Llama-2-7b-chat-hf --llama2-quant fp16 --encoder cnn3d --encoder-device cuda --pcg-refresh-period 5 --max-steps 10 --n-samples 10 --seeds 0 1 2 --n-bootstrap 2000 --output-dir outputs/E0160-smoke`
  - 运行（full，建议 1 GPU-week 预算内）：
    - `python -m provetok.experiments.fig2_scaling_multiseed --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --split test --resize-shape 64 64 64 --budget-mode flops --budgets 2000000 3000000 4000000 5000000 6000000 7000000 --costs-json outputs/compute_costs.json --pcg llama2 --llama2-path /data/models/Llama-2-7b-chat-hf --llama2-quant fp16 --encoder cnn3d --encoder-device cuda --pcg-refresh-period 5 --max-steps 20 --n-samples 231 --seeds 0 1 2 --n-bootstrap 20000 --output-dir outputs/E0160-full`
  - Status（当前）：
    - preflight（`E0160P full`）已通过：`outputs/E0160-preflight/fig2_multiseed.json`
    - full（`E0160 full`）已通过（2026-02-06 07:24 UTC）：`outputs/E0160-full/fig2_multiseed.json`

### A2. Allocation 不是“选一个方法”那么 trivial（Fig3 regret）
- [x] **E0161 (NEW)**：在多个全局配置下（例如 `b_gen/n_verify/topk`）跑 dev/test curves，让曲线交叉，再用 `regret + CI` 证明 allocation model 真有用
  - 目的：避免 Fig3 被说成“因为一个方法全程 dominate，所以 regret 当然小/无意义”。
  - Done when（最小）：
    - [x] 至少 4 个 tag 的 dev/test curves 都产出（且曲线确实交叉）
    - [x] `fig3_regret_sweep.json` 给出 `mean_normalized_regret` 的 CI，并**显著优于** naive（always_fixed_grid）
    - [ ] 口头能回答：“为什么不是后验挑配置？”（dev=val 拟合，test=test 评估）
  - 最小候选（示例 4 个 tags）：
    - `bg128_nv1_topk3`：`--b-gen 128 --n-verify 1 --topk-citations 3`
    - `bg64_nv1_topk3`：`--b-gen 64 --n-verify 1 --topk-citations 3`
    - `bg128_nv2_topk3`：`--b-gen 128 --n-verify 2 --topk-citations 3`
    - `bg128_nv1_topk1`：`--b-gen 128 --n-verify 1 --topk-citations 1`
  - 需要产物：
    - 每个 tag 一套 `baselines_curve_multiseed.json`（dev=val / test=test）
    - 合并 regret：`fig3_regret_sweep.json`
  - 运行（示例：先跑 dev=val）：
    - `python -m provetok.experiments.baselines_curve_multiseed --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --split val --resize-shape 64 64 64 --budgets 2000000 3000000 4000000 5000000 6000000 7000000 --b-gen 128 --n-verify 1 --topk-citations 3 --n-samples 100 --seeds 0 1 2 --n-bootstrap 20000 --output-dir outputs/E0161-dev-bg128_nv1_topk3`
    -（对其他 tags 重复；再跑 `--split test` 输出 test curves）
  - 运行（合并 regret）：
    - `python -m provetok.experiments.fig3_allocation_regret_sweep --metric combined --criterion bic --n-bootstrap 20000 --seed 0 --output-dir outputs/E0161-full --dev-curves bg128_nv1_topk3=outputs/E0161-dev-bg128_nv1_topk3/baselines_curve_multiseed.json bg64_nv1_topk3=outputs/E0161-dev-bg64_nv1_topk3/baselines_curve_multiseed.json bg128_nv2_topk3=outputs/E0161-dev-bg128_nv2_topk3/baselines_curve_multiseed.json bg128_nv1_topk1=outputs/E0161-dev-bg128_nv1_topk1/baselines_curve_multiseed.json --test-curves bg128_nv1_topk3=outputs/E0161-test-bg128_nv1_topk3/baselines_curve_multiseed.json bg64_nv1_topk3=outputs/E0161-test-bg64_nv1_topk3/baselines_curve_multiseed.json bg128_nv2_topk3=outputs/E0161-test-bg128_nv2_topk3/baselines_curve_multiseed.json bg128_nv1_topk1=outputs/E0161-test-bg128_nv1_topk1/baselines_curve_multiseed.json`

### A3. Citations 不是装饰：不仅“verifier 抱怨”，也要“generation correctness 下降”
- [x] **E0162 (NEW)**：`figX_counterfactual` 在真实数据上同时报告：
  - citation-nontrivial：`cite_swap` ⇒ unsupported 显著上升（Holm）
  - evidence-dependence：`evidence_drop`/`token_perm` ⇒ Frame-F1/ROUGE-L 显著下降（Holm）
  - grounding-dependence：`no_cite`/`omega_perm` ⇒ IoU_union 显著下降（Holm）
  - Done when（最小）：
    - [x] Smoke 通过：`outputs/E0162-smoke/**/figX_counterfactual.json`
    - [x] Full 通过：`outputs/E0162-full/**/figX_counterfactual.json`
    - [x] 写进 appendix：counterfactual 定义 + Holm + paired bootstrap（避免被说 p-hacking）
  - 运行（smoke）：
    - `python -m provetok.experiments.figX_counterfactual --smoke --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --split test --resize-shape 32 32 32 --flops-total 5000000 --costs-json outputs/compute_costs.json --b-gen 128 --n-verify 1 --n-samples 20 --n-bootstrap 2000 --tokenizer fixed_grid --fixed-grid-max-depth 6 --pcg-citation-strategy score --pcg-q-strategy support --topk-citations 3 --output-dir outputs/E0162-smoke`
  - 运行（full）：
    - `python -m provetok.experiments.figX_counterfactual --dataset-type manifest --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --split test --resize-shape 64 64 64 --flops-total 5000000 --costs-json outputs/compute_costs.json --b-gen 128 --n-verify 1 --n-samples 231 --n-bootstrap 20000 --tokenizer fixed_grid --fixed-grid-max-depth 6 --pcg-citation-strategy score --pcg-q-strategy support --topk-citations 3 --output-dir outputs/E0162-full`

### A4. Qualitative / 审计面板（oral 的“防猝死”）
- [x] **E0163 (NEW, low-cost)**：做 3–5 个可审计 case study（同一套可视化模板），每个 case 必须能回答：
  - “tokens/Ω 覆盖了什么区域？引用的 top-k 是哪些 cell？”
  - “verifier 到底检查了什么？unsupported/overclaim 是怎么来的？”
  - “refusal 为什么触发/不触发？有没有封嘴？”
  - Done when（最小）：
    - [x] 每个 case 输出：`case.json`（gen+issues+trace+meta）+ `case.png`（Ω+lesion overlay）
    - [ ] oral slide 里放 1 个“正例”、1 个“refusal 例”、1 个“失败例 + 可解释原因”
      - 当前已产出 `E0163-full`：`outputs/E0163-full/index.json`（共 5 个 cases）
      - 可先用作“正例/弱例”的候选：
        - 正例候选（IoU_union 较高）：`case_9404470...`（见 `outputs/E0163-full/index.json`）
        - 弱例候选（IoU_union 较低，可讲 failure mode）：`case_cc8e4ac...`（见 `outputs/E0163-full/index.json`）
      - 仍缺：明确的 refusal case（需要后续定向采样/构造）

## B) 叙事清单（oral 讲法必须闭环）

- [x] **B1 贡献 2 句**：BET（预算显式化 + scaling/allocation）+ PCG（proof-carrying + refusal calibration）（见 `docs/oral_script.md`）
- [x] **B2 主图导读 6 句**：先 Pareto/scaling，再 grounding，再 counterfactual（citations 非装饰），最后 refusal anti-gaming（见 `docs/oral_script.md`）
- [x] **B3 反质疑 4 点**：封嘴风险、verifier 强度、latency tail、跨域泛化（至少给出 1 个 failure mode + fix）（见 `docs/oral_script.md`）
- [x] **B4 Repro 1 页**：固定 commit + manifest revision + rule-set + 一键脚本 + per-sample traces（告诉 reviewer “怎么复现/怎么审计”；见 `docs/oral_script.md`）

## C) 工程与可复现（必须）

- [x] `pip install -r requirements.txt` 能跑通 `pytest -q`
- [x] 关键 runner 输出包含 `meta`（commit / data_revision / split_manifest / rule_set_version）
- [x] `.rd_queue` 可重复队列：`python scripts/rd_queue.py make && python scripts/rd_queue.py run && python scripts/rd_queue.py sync`

## D) 可选但很“加分”（仅在 P0 都过后再做）

- [ ] **D1 LLM refusal pilot**：在 `pcg=llama2` 设定下做一次 τ_refuse 的 dev→test 校准（只需要 1 budget + n=50），验证 critical miss-rate gate 不被“封嘴”绕过
- [ ] **D2 Cross-dataset sanity**：在第二个 manifest（如 CT-RATE/CT-3DRRG）上复现 E0162(smoke) 或 E0160(smoke)，只为说明“不是只对一个数据集会跑”
