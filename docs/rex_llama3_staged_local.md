# ReX llama3-only Staged Local Flow

## Stages

- `stageA`: ReX mini, `n=20`
- `stageB`: ReX mini, `n=57`
- `stageC`: ReX 100g, `n=100`
- `stageD`: ReX 100g full-like (`n=100000000`, bounded by dataset size)

Gate rule: each stage must pass `stage_check_report.py` before next stage starts.

## Local Dry-Run

```bash
bash scripts/ops/run_rex_llama3_staged_local.sh
```

Default `DRY_RUN=1`: command wiring and file outputs are validated without heavy compute.

## Real Run (Server)

```bash
DRY_RUN=0 \
LLAMA3_PATH=~/models/llama3 \
REX_MINI_MANIFEST=/data/provetok_datasets/rexgroundingct_mini/manifest.jsonl \
REX_100G_MANIFEST=/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl \
bash scripts/ops/run_rex_llama3_staged_local.sh
```

## Stage Check Outputs

Each stage writes:
- `stage_check_report.json`
- `stage_check_report.md`

Default checks:
- parse failure rate
- citation non-empty rate
- abnormal output proxy rate (`unsupported + overclaim`)
- mean warm runtime
