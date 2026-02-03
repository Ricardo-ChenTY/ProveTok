# ProveTok Project v0 (from scratch scaffold)

你现在“什么都没有”，所以这是一套**能直接运行**、并且给你预留好 Dataset/Model/Training 位置的工程骨架。

目标：先把论文式的协议闭环跑通（M0），再逐步替换为真实数据与真实模型：
**Tokenize(BET) -> Generate(PCG: frames+citations) -> Verify -> Refine**

> 当前版本用的是合成 3D volume + toy PCG + rule-based verifier，重点在**接口契约**和**可审计 artifact**。

---

## 1) 一键跑通 Demo

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

python -m provetok.run_demo --steps 3 --budget 64 --seed 0
```

会输出一个 JSON artifact（tokens/frames/citations/issues/refine_trace）。

---

## 2) 跑一个“训练脚本骨架”（当前不会训练出 SOTA，只验证 pipeline）

```bash
python scripts/train_m0.py --config configs/m0.yaml
```

它会：
1. 生成/加载 dataset（默认 synthetic）
2. 跑一遍 forward + loss（toy）
3. 保存 checkpoint 到 `./checkpoints/`

---

## 3) 你要替换的地方（预留位置）

### Dataset
- `provetok/data/dataset.py`：Dataset / DataModule 入口（现在是 synthetic）
- `provetok/data/io.py`：未来在这里接 NIfTI/DICOM 读取

### Model
- `provetok/models/encoder3d.py`：3D backbone（占位）
- `provetok/models/pcg_head.py`：frames+citations 头（占位）
- `provetok/models/system.py`：把 encoder + BET + PCG + verifier 串起来（占位）

### Training / Eval
- `scripts/train_m0.py`：训练入口（已预留 optimizer/scheduler）
- `provetok/eval/metrics_frames.py`：frames 级别 F1（占位）
- `provetok/eval/metrics_grounding.py`：grounding 指标（占位）

---

## 4) 你后续怎么扩展到真实任务（推荐顺序）
1. 用真实 CT 替换 synthetic：实现 `load_volume()` + dataset manifest
2. 用真实 3D encoder 替换 toy token embedding：改 `encode_tokens()`
3. 让 PCG 输出你定义的 frames schema，并保留 citations + verifier trace
4. 扩 verifier rules（U1/O1/I1/M1 + trace），并做预算/性能曲线

