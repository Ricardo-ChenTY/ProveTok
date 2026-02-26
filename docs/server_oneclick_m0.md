# M0 一键上服务器（双卡 A100）

## 1. 准备路径配置

```bash
cd /data/ProveTok
cp scripts/ops/server_paths.env.example scripts/ops/server_paths.env
vim scripts/ops/server_paths.env
```

至少确认这几个变量正确：

- `CT_RATE_RRG_MANIFEST`
- `LLM_PATH`
- `CT_RATE_CT2REP_ROOT`（如果你会跑 CT2Rep 链路）

## 2. 一键启动 M0（64/128/256）

```bash
cd /data/ProveTok
bash scripts/ops/oneclick_deploy_m0_server.sh
```

默认行为：

- 自动加载 `scripts/ops/server_paths.env`
- 预检 `python/torchrun/nvidia-smi`
- 预检 manifest（前 256 行：`volume_path/report_text`）
- 自动生成 runtime 配置并覆盖 `manifest_path`
- 调用现有 `scripts/ops/run_m0_multiscale_a100.sh`
- 失败即停（`STOP_ON_FAIL=1`）

## 3. 常用覆盖参数

只跑主实验 128：

```bash
TARGETS="128" bash scripts/ops/oneclick_deploy_m0_server.sh
```

切卡/进程数：

```bash
GPUS=0,1 NPROC_PER_NODE=2 bash scripts/ops/oneclick_deploy_m0_server.sh
```

临时切换 manifest（不改原始配置文件）：

```bash
MANIFEST_PATH=/data/provetok_datasets/xxx/manifest.jsonl \
bash scripts/ops/oneclick_deploy_m0_server.sh
```

关闭 preflight 或 manifest probe：

```bash
DO_PREFLIGHT=0 DO_VALIDATE=0 bash scripts/ops/oneclick_deploy_m0_server.sh
```
