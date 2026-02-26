# llama3-only Local Staged Build Manifest (No Server)

## Goal

Build a local-first, reversible orchestration layer before server runs:
- `llama3-only` CLI surface
- staged gate flow (`stageA` -> `stageD`)
- per-stage check reports
- dry-run by default for local safety

## Changed Files

- `provetok/experiments/run_baselines.py`
  - `--llama3-*` CLI arguments (removed `--llama2-*` and `--llm-path`)
  - backend choices locked to `llama3`
  - default model path to `~/models/llama3`
- `provetok/eag/__init__.py`
- `provetok/eag/llama3_eag.py`
  - lightweight llama3 naming adapter over existing generation core
- `scripts/ops/stage_check_report.py`
  - produces `stage_check_report.json` and `stage_check_report.md`
- `scripts/ops/run_rex_llama3_staged.py`
  - staged gate runner with fail-fast and dry-run support
- `scripts/ops/run_rex_llama3_staged_local.sh`
  - one-command local wrapper
- `scripts/ops/server_paths.env.example`
  - `LLAMA3_PATH=~/models/llama3`

## Explicitly Not Changed

- Core BET/EAG/verifier/refine call order and internals
- Dataset loaders and training model architecture
- R6 CT-CLIP runtime integration logic

## Local Dry-Run Quick Start

```bash
bash scripts/ops/run_rex_llama3_staged_local.sh
```

This prints planned commands and creates dry-run stage reports, without running heavy jobs.
