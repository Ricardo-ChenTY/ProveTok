# Mohu

## 1. Not Implemented

- [ ] M0200: 接入“预训练的 3D token embedding encoder”（非 toy）用于 pp.md 主叙事
  - Ref: P0103, pp.md §4.2/§6.6
  - Context: 当前 token.embedding 的 encoder 侧只有 `SimpleCNN3D`（随机初始化轻量 CNN）或 toy patch embedding；这不足以支撑“靠影像语义做可解释证据 tokenization”的强叙事，容易被 reviewer 认为是工程占位。
  - Acceptance:
    - 明确选型并接入至少一个可公开复现的预训练 encoder（优先：CT-CLIP / RadFM 二选一）；
    - `TokenEncoder(encoder=...)` 路径可直接复用该 encoder 的 feature map + ROI pooling；
    - 在 `docs/experiment.md` 增加一条 encoder ablation（smoke/full）并能跑通。
  - Verification: `python -c "from provetok.bet.tokenize import TokenEncoder; print('TokenEncoder_ok')"`

## 2. Ambiguities

- [ ] M0201: RRG-DPO 预处理已可比，但 Table1 内部 baselines 仍使用 `--resize-shape 64^3`，需要锁死“论文口径”
  - Ref: C0101, P0103, E0214, E0230, pp.md §6.2
  - Context: 数据落盘已按 RRG-DPO spacing/shape 标准化，但 `run_baselines` 为了算力会再 downsample。若 paper 表格/叙事不写清楚，审稿人会直接判为“不公平对比”。
  - Acceptance:
    - 在 Table1 的 caption/脚注写死当前 setting（downsampled vs full-res）；
    - 若要 claim strict comparability，则增加 full-res（或至少更高分辨率）的一条可跑实验，或把 claim 降级为“dev setting”。
  - Verification: `rg -n "downsample" docs/experiment.md && rg -n "resize-shape" docs/experiment.md`

## Resolved (optional)

- 历史 mohu 全量归档：`docs/mohu_legacy_20260219.md`。
