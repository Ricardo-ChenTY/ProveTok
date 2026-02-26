#!/usr/bin/env bash
set -u

# Multi-scale M0 launcher (64/128/256) with resumable strategy.
# Designed for Linux servers with multi-GPU A100.
#
# Usage:
#   bash scripts/ops/run_m0_multiscale_a100.sh
#
# Optional env overrides:
#   ROOT_DIR=/data/ProveTok
#   PYTHON_BIN=python
#   TORCHRUN_BIN=torchrun
#   GPUS=0,1
#   NPROC_PER_NODE=2
#   STAGE=M0
#   CONFIGS="configs/m0_a100.yaml configs/m0_a100_64.yaml configs/m0_a100_256.yaml"

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
STAGE="${STAGE:-M0}"
CONFIGS="${CONFIGS:-configs/m0_a100.yaml configs/m0_a100_64.yaml configs/m0_a100_256.yaml}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/outputs/_m0_multiscale_logs/${TS}"
mkdir -p "${LOG_DIR}"

get_yaml_scalar() {
  local file="$1"
  local key="$2"
  awk -F: -v k="$key" '
    $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
      v = $0
      sub(/^[^:]*:[[:space:]]*/, "", v)
      gsub(/["'"'"'[:space:]]/, "", v)
      print v
      exit
    }' "$file"
}

find_latest_ckpt() {
  local stage_dir="$1"
  if [ -f "${stage_dir}/ckpt_final.pt" ]; then
    echo "${stage_dir}/ckpt_final.pt"
    return 0
  fi
  local latest
  latest="$(ls -1 "${stage_dir}"/ckpt_step*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "${latest}" ]; then
    echo "${latest}"
  fi
}

run_one() {
  local cfg="$1"
  local name
  name="$(basename "${cfg}" .yaml)"
  local log_file="${LOG_DIR}/${name}.log"

  if [ ! -f "${cfg}" ]; then
    echo "[SKIP] missing config: ${cfg}" | tee -a "${log_file}"
    return 0
  fi

  local run_name save_dir stage_dir latest
  run_name="$(get_yaml_scalar "${cfg}" "run_name")"
  save_dir="$(get_yaml_scalar "${cfg}" "save_dir")"
  if [ -z "${run_name}" ]; then
    run_name="${name}"
  fi
  if [ -z "${save_dir}" ]; then
    save_dir="outputs"
  fi
  if [[ "${save_dir}" != /* ]]; then
    stage_dir="${ROOT_DIR}/${save_dir}/${run_name}/${STAGE}"
  else
    stage_dir="${save_dir}/${run_name}/${STAGE}"
  fi

  mkdir -p "${stage_dir}"
  latest="$(find_latest_ckpt "${stage_dir}")"

  if [ "${latest}" = "${stage_dir}/ckpt_final.pt" ]; then
    echo "[SKIP] ${cfg} already finished (${latest})" | tee -a "${log_file}"
    return 0
  fi

  echo "[START] ${cfg}" | tee -a "${log_file}"
  echo "  stage_dir=${stage_dir}" | tee -a "${log_file}"
  if [ -n "${latest}" ]; then
    echo "  resume_from=${latest}" | tee -a "${log_file}"
  else
    echo "  resume_from=<none>" | tee -a "${log_file}"
  fi

  CUDA_VISIBLE_DEVICES="${GPUS}" \
    "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train_m0.py --config "${cfg}" --stage "${STAGE}" --auto-resume >>"${log_file}" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then
    echo "[DONE ] ${cfg}" | tee -a "${log_file}"
  else
    echo "[FAIL ] ${cfg} rc=${rc}" | tee -a "${log_file}"
  fi
  return ${rc}
}

cd "${ROOT_DIR}" || exit 1
echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] gpus=${GPUS} nproc=${NPROC_PER_NODE}"
echo "[INFO] configs=${CONFIGS}"

for cfg in ${CONFIGS}; do
  if ! run_one "${cfg}"; then
    if [ "${STOP_ON_FAIL}" = "1" ]; then
      echo "[ABORT] stop on first failure because STOP_ON_FAIL=1"
      exit 1
    fi
  fi
done

echo "[INFO] all requested configs finished. logs at ${LOG_DIR}"
