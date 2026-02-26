#!/usr/bin/env bash
set -euo pipefail

# Local cautious staged runner (no server dependency by default).
# Default mode is dry-run so it only validates command wiring and paths.
#
# Example:
#   bash scripts/ops/run_rex_llama3_staged_local.sh
#   DRY_RUN=0 REX_MINI_MANIFEST=/data/.../mini.jsonl REX_100G_MANIFEST=/data/.../100g.jsonl \
#   LLAMA3_PATH=~/models/llama3 bash scripts/ops/run_rex_llama3_staged_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DRY_RUN="${DRY_RUN:-1}"
START_FROM="${START_FROM:-stageA}"
ONLY_STAGE="${ONLY_STAGE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/rex_llama3_staged}"

LLAMA3_PATH="${LLAMA3_PATH:-~/models/llama3}"
LLAMA3_QUANT="${LLAMA3_QUANT:-fp16}"
LLAMA3_CONTRACT_MODE="${LLAMA3_CONTRACT_MODE:-full}"
LLAMA3_CITATION_SOURCE="${LLAMA3_CITATION_SOURCE:-score_override}"
LLAMA3_MAX_FRAMES="${LLAMA3_MAX_FRAMES:-1}"
LLAMA3_LORA_ADAPTER="${LLAMA3_LORA_ADAPTER:-}"
LLAMA3_LORA_MERGE="${LLAMA3_LORA_MERGE:-0}"

REX_MINI_MANIFEST="${REX_MINI_MANIFEST:-/data/provetok_datasets/rexgroundingct_mini/manifest.jsonl}"
REX_100G_MANIFEST="${REX_100G_MANIFEST:-/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl}"
SEED="${SEED:-42}"
TOPK_CITATIONS="${TOPK_CITATIONS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}" "scripts/ops/run_rex_llama3_staged.py"
  "--start-from" "${START_FROM}"
  "--output-root" "${OUTPUT_ROOT}"
  "--llama3-path" "${LLAMA3_PATH}"
  "--llama3-quant" "${LLAMA3_QUANT}"
  "--llama3-contract-mode" "${LLAMA3_CONTRACT_MODE}"
  "--llama3-citation-source" "${LLAMA3_CITATION_SOURCE}"
  "--llama3-max-frames" "${LLAMA3_MAX_FRAMES}"
  "--rex-mini-manifest" "${REX_MINI_MANIFEST}"
  "--rex-100g-manifest" "${REX_100G_MANIFEST}"
  "--seed" "${SEED}"
  "--topk-citations" "${TOPK_CITATIONS}"
)

if [ -n "${ONLY_STAGE}" ]; then
  CMD+=("--only-stage" "${ONLY_STAGE}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=("--dry-run")
fi
if [ -n "${LLAMA3_LORA_ADAPTER}" ]; then
  CMD+=("--llama3-lora-adapter" "${LLAMA3_LORA_ADAPTER}")
fi
if [ "${LLAMA3_LORA_MERGE}" = "1" ]; then
  CMD+=("--llama3-lora-merge")
fi
if [ -n "${EXTRA_ARGS}" ]; then
  CMD+=("--extra-args" "${EXTRA_ARGS}")
fi

echo "[INFO] repo=${ROOT_DIR}"
echo "[INFO] dry_run=${DRY_RUN} output=${OUTPUT_ROOT}"
echo "[INFO] llama3_path=${LLAMA3_PATH}"
echo "[INFO] rex_mini_manifest=${REX_MINI_MANIFEST}"
echo "[INFO] rex_100g_manifest=${REX_100G_MANIFEST}"

"${CMD[@]}"

