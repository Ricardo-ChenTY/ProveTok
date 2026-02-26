#!/usr/bin/env bash
set -euo pipefail

# One-click server launcher for M0 multi-scale runs (64/128/256).
# It only wraps existing pipeline scripts; no core workflow changes.
#
# Quick usage:
#   cp scripts/ops/server_paths.env.example scripts/ops/server_paths.env
#   # edit scripts/ops/server_paths.env if needed
#   bash scripts/ops/oneclick_deploy_m0_server.sh
#
# Optional env overrides:
#   PATHS_ENV=/data/ProveTok/scripts/ops/server_paths.env
#   ROOT_DIR=/data/ProveTok
#   PYTHON_BIN=python
#   TORCHRUN_BIN=torchrun
#   GPUS=0
#   NPROC_PER_NODE=1
#   TARGETS="128 64 256"   # or "128"
#   MANIFEST_PATH=/data/.../manifest_rrg_dpo.jsonl
#   DO_PREFLIGHT=1
#   DO_VALIDATE=1
#   PROBE_LINES=256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATHS_ENV="${PATHS_ENV:-${ROOT_DIR}/scripts/ops/server_paths.env}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
GPUS="${GPUS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
STAGE="${STAGE:-M0}"
TARGETS="${TARGETS:-128 64 256}"

DO_PREFLIGHT="${DO_PREFLIGHT:-1}"
DO_VALIDATE="${DO_VALIDATE:-1}"
PROBE_LINES="${PROBE_LINES:-256}"

require_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "[ERR ] missing command: ${c}"
    exit 1
  fi
}

require_file() {
  local p="$1"
  if [ ! -f "${p}" ]; then
    echo "[ERR ] missing file: ${p}"
    exit 1
  fi
}

if [ ! -f "${PATHS_ENV}" ]; then
  if [ -f "${ROOT_DIR}/scripts/ops/server_paths.env.example" ]; then
    cp "${ROOT_DIR}/scripts/ops/server_paths.env.example" "${PATHS_ENV}"
    echo "[ERR ] ${PATHS_ENV} not found. Template copied from .example. Fill it then rerun."
    exit 1
  fi
  echo "[ERR ] paths env missing: ${PATHS_ENV}"
  exit 1
fi

# shellcheck disable=SC1090
source "${PATHS_ENV}"

MANIFEST_PATH="${MANIFEST_PATH:-${CT_RATE_RRG_MANIFEST:-/data/provetok_datasets/ct_rate_100g_rrg_dpo_all/manifest_rrg_dpo.jsonl}}"

CFG_128="${ROOT_DIR}/configs/m0_a100.yaml"
CFG_64="${ROOT_DIR}/configs/m0_a100_64.yaml"
CFG_256="${ROOT_DIR}/configs/m0_a100_256.yaml"

if [ "${DO_PREFLIGHT}" = "1" ]; then
  echo "[INFO] running preflight checks..."
  require_cmd "${PYTHON_BIN}"
  require_cmd "${TORCHRUN_BIN}"
  require_cmd nvidia-smi
  require_file "${CFG_128}"
  require_file "${CFG_64}"
  require_file "${CFG_256}"
  require_file "${MANIFEST_PATH}"

  nvidia-smi -L || true
  "${PYTHON_BIN}" - <<'PY'
import torch
print(f"[INFO] torch={torch.__version__}, cuda={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}")
PY
fi

if [ "${DO_VALIDATE}" = "1" ]; then
  echo "[INFO] probing manifest contract on first ${PROBE_LINES} rows..."
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PROBE_LINES}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
max_rows = int(sys.argv[2])
if not manifest.exists():
    raise SystemExit(f"[ERR ] manifest missing: {manifest}")

n = 0
bad_json = 0
bad_volume = 0
bad_report = 0
with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        n += 1
        try:
            obj = json.loads(s)
        except Exception:
            bad_json += 1
            if n >= max_rows:
                break
            continue
        if not str(obj.get("volume_path", obj.get("volume", "")) or "").strip():
            bad_volume += 1
        if not str(obj.get("report_text", obj.get("report", "")) or "").strip():
            bad_report += 1
        if n >= max_rows:
            break

if n == 0:
    raise SystemExit("[ERR ] manifest is empty")
if bad_json or bad_volume or bad_report:
    raise SystemExit(
        f"[ERR ] manifest probe failed: n={n}, bad_json={bad_json}, "
        f"bad_volume={bad_volume}, bad_report={bad_report}"
    )
print(f"[OK  ] manifest probe passed: n={n}, path={manifest}")
PY
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUNTIME_CFG_DIR="${ROOT_DIR}/outputs/_runtime_configs/${TS}"
mkdir -p "${RUNTIME_CFG_DIR}"

render_cfg() {
  local src="$1"
  local dst="$2"
  sed -E "s|^([[:space:]]*manifest_path:).*|\1 \"${MANIFEST_PATH}\"  # runtime override by oneclick script|g" "${src}" > "${dst}"
}

CONFIGS=""
for t in ${TARGETS}; do
  case "${t}" in
    128)
      out="${RUNTIME_CFG_DIR}/m0_a100.yaml"
      render_cfg "${CFG_128}" "${out}"
      CONFIGS="${CONFIGS} ${out}"
      ;;
    64)
      out="${RUNTIME_CFG_DIR}/m0_a100_64.yaml"
      render_cfg "${CFG_64}" "${out}"
      CONFIGS="${CONFIGS} ${out}"
      ;;
    256)
      out="${RUNTIME_CFG_DIR}/m0_a100_256.yaml"
      render_cfg "${CFG_256}" "${out}"
      CONFIGS="${CONFIGS} ${out}"
      ;;
    *)
      echo "[ERR ] unsupported target in TARGETS: ${t} (allowed: 64 128 256)"
      exit 1
      ;;
  esac
done

CONFIGS="$(echo "${CONFIGS}" | xargs)"
if [ -z "${CONFIGS}" ]; then
  echo "[ERR ] no configs selected. Set TARGETS to one or more of: 64 128 256"
  exit 1
fi

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] paths_env=${PATHS_ENV}"
echo "[INFO] manifest=${MANIFEST_PATH}"
echo "[INFO] gpus=${GPUS} nproc=${NPROC_PER_NODE}"
echo "[INFO] targets=${TARGETS}"
echo "[INFO] runtime_cfg_dir=${RUNTIME_CFG_DIR}"

STOP_ON_FAIL=1 \
ROOT_DIR="${ROOT_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
TORCHRUN_BIN="${TORCHRUN_BIN}" \
GPUS="${GPUS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
STAGE="${STAGE}" \
CONFIGS="${CONFIGS}" \
bash "${ROOT_DIR}/scripts/ops/run_m0_multiscale_a100.sh"

echo "[OK  ] one-click M0 deploy finished."
