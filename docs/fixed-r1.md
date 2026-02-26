# ProveTok 阶段性工程重构与 MVR 修订并行执行计划  
**文档版本:** v1.1 (2026-02-25)  
**当前分支:** `fix-r1-metrics`  
**目标设定:** 支撑 MICCAI 2026 / ECCV 2026 论文的大规模重制，对齐 $128^3$ 分辨率与多模态审计评价体系。

---

## 第一部分：M0 阶段分布式训练基建升级报告 (已部署)

为了响应审稿人关于“64³ 分辨率下病灶尺寸过小导致空间 Grounding 失去临床意义”的致命批评，我们已将主实验分辨率提升至 $128^3$。此分辨率的提升对 GPU 显存与 CPU I/O 提出了指数级增长的要求。为此，我们对底层训练流水线进行了以下工业级重构：

### 1. 核心分布式训练引擎重构 (`provetok/training/trainer.py`)

**动态计算图与 DDP 不兼容问题：**  
由于 ProveTok 独有的 BET (Budgeted Evidence Tokenization) 架构会产生动态的 Octree Token 序列，在 `_train_step` 中必须使用 `for b in range(B)` 逐样本进行确定性前向传播。这导致标准 PyTorch DistributedDataParallel (DDP) 在计算图中多次 Forward 而无法正确同步梯度。

**计算图兼容与混合精度：**  
全面引入 Hugging Face Accelerator 替代原生 DDP，并强制挂载 `mixed_precision="bf16"`。该方案兼容动态计算图的梯度累加 (`accelerator.backward(total_loss)`)，并自动处理 BF16 下的梯度缩放与裁剪 (`accelerator.clip_grad_norm_`)。

**进程隔离与状态安全：**  
在模型验证 (`_eval_step`)、日志记录与检查点持久化 (`_save_checkpoint`) 阶段，严格部署 `self.accelerator.is_local_main_process` 屏障。

**模型拓扑纯净化：**  
在保存权重前，强制调用 `accelerator.unwrap_model()` 对模型进行解包，防止 DDP 在 `state_dict` 中注入 `module.` 前缀，确保后续 M2/M3 阶段 LLM 微调时的单卡权重无缝加载。

---

### 2. 数据流水线与 I/O 吞吐解除瓶颈 (`scripts/train_m0.py`)

**路径透传与真实数据挂载：**  
移除了原代码中针对 Smoke Test 的硬编码（`type: synthetic` 无法读取外部路径）。通过配置字典重构，显式暴露 `manifest_path` 参数，成功将 660GB 的 CT-RATE 真实数据清单挂载至底层 `dataset.py`。

**并行解压与 GPU 饥饿防御：**  
NIFTI 医学图像在加载时存在巨大的 CPU 解压开销。通过从配置文件动态读取 `num_workers`，彻底摒弃单线程阻塞式读取（`num_workers: 0`），有效保障双卡 A100 的计算利用率。

---

### 3. 生产级双卡超参定型 (`configs/m0_a100.yaml`)

- 数据规格：启用 `type: real`，分辨率锁定为 `vol_shape: [128, 128, 128]`  
- 批次策略：单卡 `batch_size: 8`，双卡等效 Batch Size = 16  
- 并发读取线程：`num_workers: 16`

---

## 第二部分：MVR (最小可行修订) 冲刺期并行开发路线图

在双卡 A100 进行 M0 阶段耗时跑批的窗口期内，本地算法开发资源必须立即转移至论文评价体系的补齐（MVR 规划），以确保模型完成训练后能够实现自动化一键评测。

---

## 📌 优先级 1：核心防御指标系统开发 (MVR #2) — *当前待办*

### 背景驱动
原论文在 Open-loop 模式下 `AnyIssue=0` 被审稿人指出存在同义反复（Tautology），且系统极易通过“放弃诊断（Abstention）”刷高合规率。CheXbert 因模态不匹配（X光 vs 3D CT）导致结果失真。

### 开发计划

**新建 RadBERT 判别器模块 (`provetok/eval/metrics_radbert.py`)：**  
- 引入 CT-RATE 官方推荐的 RadBERT（18 类）Label Extractor  
- 使用 `transformers` 封装为批量推理的二分类判定器  
- 输出维度：$N \times 18$

**评估主轴集成 (`scripts/paper/compute_paper_metrics.py`)：**  
- 引入 `abstention_rate`（弃权率）规则（检测输出长度阈值、"abstain"/"cannot determine" 等兜底短语）  
- 在 Batch 层面引入 RadBERT 特征  
- 计算 `finding_precision`、`finding_recall`、`finding_f1`  
- 生成 3D Pareto 曲面数据

---

## 📌 优先级 2：同代 SOTA 基线评测对齐 (MVR #3)

### 背景驱动
R1 审稿人指出缺乏与 Reg2RG、BTB3D 等同期核心工作的对比。

### 开发计划
- 编写独立推理脚本：`scripts/external/run_reg2rg_baseline.py`  
- 调用 Reg2RG 开源代码与 Checkpoint  
- 使用相同 CT-RATE 测试集（$N=137$ 与 $N=1304$ 验证集）  
- 输出标准化 `.jsonl` 预测  
- 送入优先级 1 的评估主轴，提前填充论文 Table 1

---

## 📌 优先级 3：临床先验验证资源排期 (Human Evaluation)

### 开发计划
- 制定盲评操作协议（Blinded Protocol）  
- 设计临床打分量表：  
  - Spatial Relevance (1–5)  
  - Diagnostic Utility (1–5)  
  - Laterality Correctness (Binary)  
- 建立与 2 位 Board-certified 放射科医生的 50-case 流转通道

---

## 📌 优先级 4：M2/M3 架构完整版升级筹备 (Full Revision)

等待 M0 预训练收敛后启动：

- **语义逻辑校验器上线：**  
  - 基于 TotalSegmentator 的解剖学错位规则 (R5 AnatomyMismatch)  
  - 基于 CT-CLIP 的跨模态语义相关性规则 (R6 SemanticRelevance)

- **LLM 内联引用微调：**  
  - 主干升级至 LLaMA-3.1-8B-Instruct  
  - 挂载 RadGenome 624K 监督数据  
  - 训练端到端的 `[CIT_N]` 标签生成能力

---

# GeAR: Mitigating Negative RAG Gain via Training-Free Confidence Gating
