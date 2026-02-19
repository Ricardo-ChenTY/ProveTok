# ReXrank Local Verification (Gold Masks Available Locally)

本文件用于回答一个很具体的问题：

> ReXGroundingCT 的 train/val 其实是 gold voxel masks。我们当前的 ReXrank submission pipeline 在这些可见 gold 上“到底有没有对上”？如果很差，那就不要把 hidden test submission 当成口头/论文的强证据。

注意：这不是 hidden test 的官方分数（hidden test 的 GT 不公开）；这里只做**本地可见 gold**（我们已有 manifest 里 `mask_path` 存在的子集）上的 sanity check。

## What We Ran

脚本：`scripts/external/verify_rexrank_manifest.py`

共同配置：

- saliency weights：`outputs/E0155-train_saliency_cnn3d_100g/saliency_cnn3d.pt`
- resize：`64 64 64`（SaliencyCNN3D 输入）
- `mask_ratio=0.005`
- `min_size=50`, `connectivity=2`
- `global_hit_thr=0.1`（与官方 `rexrank_eval.py` 的 GLOBAL_HIT_THR 一致）
- methods：
  - `components_greedy`（与 E0190 submission 默认一致：从 union mask 拆 CC，再按 laterality/上下叶启发式分配到 finding）
  - `replicate`（对照：把 union mask 复制到所有 finding channel）

命令（已执行）：

- mini：`python scripts/external/verify_rexrank_manifest.py --manifest /data/provetok_datasets/rexgroundingct_mini/manifest.jsonl --splits val test --out-dir outputs/E0192-rexrank-manifest-verify-mini --device cuda --methods components_greedy replicate`
- 100g：`python scripts/external/verify_rexrank_manifest.py --manifest /data/provetok_datasets/rexgroundingct_100g/manifest.jsonl --splits val test --out-dir outputs/E0192-rexrank-manifest-verify-100g --device cuda --methods components_greedy replicate`

产物（不入 git）：`outputs/E0192-rexrank-manifest-verify-*/verify_rexrank_manifest.json`

补充（2026-02-10）：我们新增了更强的 saliency 后端并保持 submission 协议不变（同 `mask_ratio/min_size/connectivity`），用于回答 “把本地可见 gold 的 hit/dice 拉到不退化的量级”：

- saliency weights：`outputs/E0201-train_saliency_unet3d_posw100_100g/saliency_unet3d.pt`（UNet3D + bce_pos_weight clamp）
- 验证（smoke, test 前 50 cases）：`outputs/E0201-verify_rexrank_manifest_unet3d_posw100_test50/verify_rexrank_manifest.json`
- 验证（full, test 全量 231 cases）：`outputs/E0206-verify_rexrank_manifest_unetposw100_100g_test_full/verify_rexrank_manifest.json`

## Results (Finding-Weighted)

说明：finding-weighted 表示每个 finding channel 等权；`hit_dice` 是 `Dice >= 0.1` 的比例。

### UNet3D (pos_weight) on `rexgroundingct_100g` (test, smoke)

从 `outputs/E0201-verify_rexrank_manifest_unet3d_posw100_test50/verify_rexrank_manifest.json` 读取（test 前 50 cases）：

- test (50 cases / 130 findings)
  - `components_greedy`: Dice=0.0159, IoU=0.0112, hit_dice=0.038, hit_any=0.185

从 `outputs/E0206-verify_rexrank_manifest_unetposw100_100g_test_full/verify_rexrank_manifest.json` 读取（test 全量 231 cases）：

- test (231 cases / 545 findings)
  - `components_greedy`: Dice=0.0153, IoU=0.0097, hit_dice=0.048, hit_any=0.141

从 `outputs/E0207-verify_rexrank_manifest_unetposw100_topk3_test_full/verify_rexrank_manifest.json` 读取（test 全量 231 cases；multi-component assignment）：

- test (231 cases / 545 findings)
  - `components_topk` (topk=3): Dice=0.0198, IoU=0.0118, hit_dice=0.072, hit_any=0.250

### SaliencyCNN3D on `rexgroundingct_mini` (val/test)

- test (57 cases / 113 findings)
  - `components_greedy`: Dice=0.0022, IoU=0.0011, hit_dice=0.000, hit_any=0.088
  - `replicate`: Dice=0.0036, IoU=0.0018, hit_dice=0.000, hit_any=0.319
- val (60 cases / 129 findings)
  - `components_greedy`: Dice=0.0030, IoU=0.0016, hit_dice=0.000, hit_any=0.078
  - `replicate`: Dice=0.0046, IoU=0.0024, hit_dice=0.000, hit_any=0.302

### SaliencyCNN3D on `rexgroundingct_100g` (val/test)

- test (231 cases / 545 findings)
  - `components_greedy`: Dice=0.0026, IoU=0.0014, hit_dice=0.006, hit_any=0.088
  - `replicate`: Dice=0.0048, IoU=0.0024, hit_dice=0.006, hit_any=0.323
- val (55 cases / 158 findings)
  - `components_greedy`: Dice=0.0026, IoU=0.0013, hit_dice=0.006, hit_any=0.089
  - `replicate`: Dice=0.0039, IoU=0.0020, hit_dice=0.000, hit_any=0.348

## Interpretation (Actionable)

- 以 `global_hit_thr=0.1`（官方脚本默认）衡量时：
  - 旧后端（SaliencyCNN3D）在本地可见 gold 上的 `hit_dice` 仍很低（`~0.0%~0.6%`，见上方 E0192 结果）。
  - 新后端（SaliencyUNet3D + pos_weight）在 `rexgroundingct_100g` 的 test 全量上已明显改善（finding-weighted `hit_dice≈4.8%`、`hit_any≈14.1%`；Dice≈0.0153）。
  - 进一步用 `components_topk(topk=3)` 做 multi-component 分配后，test 全量达到 `hit_dice≈7.2%`、`hit_any≈25.0%`（Dice≈0.0198），说明 “一 finding 只取 1 个 component” 是主要瓶颈之一。
- `replicate` 明显比 `components_greedy` 更容易命中（`hit_any` 更高），说明当前的 “union→CC→按文字拆分到 finding” 启发式在很多 case 上会把少量正确体素分配错 channel 或直接被 `min_size` 过滤掉。
- 因此：
  - **不要把 E0190 的 hidden test submission 当作强证据**（最多说“提交通道跑通”）。
  - 若要把 hidden test 作为外部 gold 证据：本地可见 gold 的 hit/dice 需要长期保持“非退化”（当前已达成），并最好进一步提高（特别是 `hit_dice`），同时应报告 `replicate` vs `components_greedy` 的差距以量化“分配启发式”的上限。

下一步最直接的提升方向（按收益/代价排序）：

1. 调参：增大 `mask_ratio`（例如 `0.01~0.05`）并降低/关闭 `min_size` 过滤；同时在本脚本上做 val/test sweep。
2. 改分配：允许每个 finding 取 Top-K components（不是 1 个），并加入 “finding 数量 vs components 数量” 的稳健策略。（已落地 `components_topk`；见 E0207）
3. 换模型：用 text-conditioned / phrase-grounded 的分割后端（而不是 union saliency 再拆分）。
