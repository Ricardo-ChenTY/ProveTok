# experiment 分支改名后的改动记录（截至 2026-02-25）

本文记录分支改为 `experiment` 后，本地已完成的主要改动，便于后续按 round 继续推进。

## 1. 数据集与数据入口

- 新增 CT-RATE manifest 构建脚本：`scripts/data/build_ct_rate_manifest.py`
  - 支持从 CT 文件目录 + 报告表生成 `manifest.jsonl`
  - 输出 split 信息、元数据与重复报告检查结果
  - 可作为后续 RRG-DPO 预处理的上游入口
- 更新 `.gitignore`
  - 放行 `scripts/data/build_ct_rate_manifest.py` 以便纳入版本管理
- 更新 `README.md`
  - 增加“从原始 CT-RATE 数据构建 manifest”的使用说明
- 更新 `tests/test_cli_help.py`
  - 增加 `build_ct_rate_manifest.py --help` 的 CLI 冒烟测试

## 2. R1 方向（按 proposal 的 round-by-round 修复）

- 更新 `scripts/paper/compute_paper_metrics.py`
  - 新增 finding 级代理指标（基于 report 文本抽取）：
    - `finding_precision`
    - `finding_recall`
    - `finding_f1`
    - `abstention_rate`
  - 用于先补齐 R1-2（Finding Recall / Abstention）的一轮可运行口径
  - 同时补了 mixed-layout 导入兼容（`provetok` 与 `ProveTok` 路径）
- 更新 `scripts/paper/run_table1_ct_rate.py`
  - `--run-proof-external` 场景下默认 `--resize-shape` 从 `64^3` 调为 `128^3`
  - 对齐 R1 主实验分辨率口径（`128^3` 主，`64/256` 做 ablation）
  - 同步补了 mixed-layout 导入兼容
- 更新 `docs/experiment.md`
  - 明确写入 R1 分辨率策略：`128^3` 主口径，`64^3/256^3` 为 ablation
  - E0214 命令示例同步到 `--resize-shape 128 128 128`
- 更新 `README.md`
  - 同步 R1 分辨率口径与上述 finding 级代理指标说明

## 3. 大规模运行编排（双 A100）

- 新增运行脚本：`scripts/ops/launch_all_datasets_dual_a100.sh`
  - 面向 Linux 服务器，按双卡（默认 `GPU0/GPU1`）并行挂任务
  - 将最大数据集（CT-RATE）设为高优先级
  - 集成 CT-RATE 预处理、CT2Rep train/infer、多数据集 LLM 跑批、Table1 汇总
  - 关键路径和 GPU 号均可通过环境变量覆盖

## 4. 当前状态说明

- 当前工作区中仍有其他未提交改动与未跟踪文件（含实验产物/临时目录）
- 本记录文件用于说明“改名后已做了什么”，不改变既有实验结论

