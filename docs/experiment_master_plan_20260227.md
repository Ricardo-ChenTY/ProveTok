# ProveTok 实验总目标与执行路线（Server Runbook）

文档日期：2026-02-27  
目标分支：`single-card-exp`  
当前策略：先完成 MVR 可复现实验链（R6 暂缓），再补完整版修订项。

## 1. 总目标（论文导向）

围绕修订提案，形成一条可复现、可审计的实验主线：

1. 以 `128^3` 为主实验分辨率，补齐 `64^3/256^3` 消融。
2. 在 budget 约束下评估 grounded generation 与 verifier 合规表现。
3. 报告核心防御指标，避免只看 `AnyIssue` 的同义反复风险。
4. 输出可直接进入论文表格/图的数据产物与日志。

## 2. 实验阶段定义（按提案）

## Stage 1: BET（Evidence Tokenization）

- 主实验：`128^3`
- 消融：`64^3`、`256^3`
- budget 网格：`B in {64, 128, 256, 512, 1024}`
- 默认：variance-based saliency（learned saliency 作为 ablation）

当前执行要求：
- 优先保证三套分辨率配置可跑、可复现、可断点续跑。

## Stage 2: Grounded Report Generation（PCG/EAG）

- 统一入口：`llama3-only`
- 模型路径：`~/models/llama3`（服务器可改为绝对路径）
- 先跑通现有链路，产出可比较结果

说明：
- 提案完整版中的“inline citation 端到端训练（[CIT_xxx] + L_text/L_cite/L_ground）”属于后续补强项。

## Stage 3: Multi-Level Verification

- 当前主线：先稳定跑 `R1-R4`
- `R5`：数据契约/钩子可选启用（不破坏主流程）
- `R6`：CT-CLIP 语义相关性规则暂缓到下一阶段

## Stage 4: Closed-Loop Refinement

- 保持现有 refine 主链可运行
- 先保证 violation 场景下的流程可执行与可记录
- R5/R6 rerank 闭环、R6 scorer 集成属于后续增强

## 3. 当前分支已落地能力（single-card-exp）

已完成：

1. `llama3-only` 参数接口（移除 `llama2/llm-path` 对外入口）
2. staged gate 编排：
   - `stageA`（ReX mini n=20）
   - `stageB`（ReX mini n=57）
   - `stageC`（ReX 100g n=100）
   - `stageD`（ReX 100g 全量）
3. 每阶段自动检查报告：
   - `stage_check_report.json`
   - `stage_check_report.md`
4. server 路径模板中统一 `LLAMA3_PATH`

关键脚本：

- `scripts/ops/run_rex_llama3_staged_local.sh`
- `scripts/ops/run_rex_llama3_staged.py`
- `scripts/ops/stage_check_report.py`
- `provetok/experiments/run_baselines.py`

## 4. 明确未完成项（非明日阻塞）

1. `R6` CT-CLIP scorer 与阈值标定
2. Stage2 inline citation 端到端训练路径
3. R5/R6 violation 后的 citation rerank 闭环
4. 完整外部 baseline 对齐与人工评估执行

## 5. Server 执行步骤（可直接照跑）

## 5.1 同步代码

```bash
cd /data/ProveTok
git fetch origin
git checkout single-card-exp
git pull --ff-only origin single-card-exp
git rev-parse --short HEAD
```

## 5.2 环境与模型

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

下载模型到约定路径（示例）：

```bash
huggingface-cli login
mkdir -p ~/models/llama3
```

## 5.3 数据与 manifest

必须保证存在可用 manifest（至少先有 ReX mini/可替代同源 manifest）：

- `REX_MINI_MANIFEST`
- `REX_100G_MANIFEST`

若无 manifest，先生成并校验：

```bash
python scripts/data/build_rex_groundingct_manifest.py --help
python scripts/data/validate_manifest.py --help
```

## 5.4 配置路径

```bash
cp scripts/ops/server_paths.env.example scripts/ops/server_paths.env
source scripts/ops/server_paths.env
```

至少确认三项：

- `LLAMA3_PATH`
- `REX_MINI_MANIFEST`
- `REX_100G_MANIFEST`

## 5.5 先 dry-run 再正式分阶段

```bash
DRY_RUN=1 bash scripts/ops/run_rex_llama3_staged_local.sh
```

然后按阶段执行：

```bash
DRY_RUN=0 ONLY_STAGE=stageA bash scripts/ops/run_rex_llama3_staged_local.sh
DRY_RUN=0 ONLY_STAGE=stageB bash scripts/ops/run_rex_llama3_staged_local.sh
DRY_RUN=0 ONLY_STAGE=stageC bash scripts/ops/run_rex_llama3_staged_local.sh
DRY_RUN=0 ONLY_STAGE=stageD bash scripts/ops/run_rex_llama3_staged_local.sh
```

## 5.6 并行执行 Stage1 分辨率实验

使用现有配置：

- `configs/m0_a100.yaml`（128）
- `configs/m0_a100_64.yaml`
- `configs/m0_a100_256.yaml`

按资源计划串行或并行运行，保留日志与 checkpoint。

## 6. 阶段通过标准（Gate）

每个阶段至少检查：

1. 产物完整：`baselines.json` / `stage_check_report.json` / 日志存在
2. 结构健康：解析失败率、citation 非空率、异常输出率在阈值内
3. 资源可控：时长与失败样本可解释

只要某阶段 `overall_pass=false`，先修复再进入下一阶段。

## 7. 指标与汇报最小集合（MVR）

明日最小可交付指标：

1. `Finding Recall`
2. `Abstention Rate`
3. `AnyIssue`（配合 Recall 解读）
4. grounding 相关指标（IoU/hit/coverage 中至少一组）

结论优先级：
- 先确认“能诊断 + 能引用 + 能过规则”，再优化绝对分数。

## 8. 风险与回滚策略

1. 若某阶段失败：停在当前阶段，保留产物，定位 manifest/路径/模型问题。
2. 不在 server 临时改主流程代码，优先改 env 和运行参数。
3. 所有新增实验配置/脚本先 dry-run，再正式执行。

## 9. 一句话执行原则

先把 MVR 主线完整跑通并稳定出表，再进入 R6 与 inline citation 的完整版升级。
