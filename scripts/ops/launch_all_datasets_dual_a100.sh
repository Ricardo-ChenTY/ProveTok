#!/usr/bin/env bash
set -u

# Run ProveTok experiments across datasets with 2x A100 (80G), prioritizing CT-RATE.
# This script is designed for Linux servers.
#
# Usage:
#   bash scripts/ops/launch_all_datasets_dual_a100.sh
#
# Optional env overrides:
#   ROOT_DIR=/data/ProveTok
#   PYTHON_BIN=python
#   GPU0=0 GPU1=1
#   LLM_PATH=/data/models/Meta-Llama-3.1-8B-Instruct
#   PCG_BACKEND=llama3
#   RUN_RADEVAL=0

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_RADEVAL="${RUN_RADEVAL:-0}"
PCG_BACKEND="${PCG_BACKEND:-llama3}"

LOG_DIR="${ROOT_DIR}/outputs/_all_dataset_launch_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---------------------------
# Dataset/model paths (edit these for your server)
# ---------------------------
LLM_PATH="${LLM_PATH:-${LLAMA2_PATH:-/data/models/Meta-Llama-3.1-8B-Instruct}}"

CT_RATE_RAW_MANIFEST="${CT_RATE_RAW_MANIFEST:-/data/provetok_datasets/ct_rate_100g/manifest.jsonl}"
CT_RATE_RRG_ROOT="${CT_RATE_RRG_ROOT:-/data/provetok_datasets/ct_rate_100g_rrg_dpo_all}"
CT_RATE_RRG_MANIFEST="${CT_RATE_RRG_MANIFEST:-/data/provetok_datasets/ct_rate_100g_rrg_dpo_all/manifest_rrg_dpo.jsonl}"
CT_RATE_TEST_MANIFEST="${CT_RATE_TEST_MANIFEST:-/data/provetok_datasets/ct_rate_100g_rrg_dpo_test/manifest_rrg_dpo.jsonl}"
CT_RATE_CT2REP_ROOT="${CT_RATE_CT2REP_ROOT:-/data/provetok_datasets/ct_rate_100g_rrg_dpo_all_ct2rep}"

REX_MINI_MANIFEST="${REX_MINI_MANIFEST:-/data/provetok_datasets/rexgroundingct_mini/manifest.jsonl}"
REX_100G_MANIFEST="${REX_100G_MANIFEST:-/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl}"
RADGENOME_MANIFEST="${RADGENOME_MANIFEST:-/data/provetok_datasets/radgenome_chestct/manifest.jsonl}"

CT2REP_RUN_DIR="${CT2REP_RUN_DIR:-/data/provetok_runs/ct2rep_ct_rate_100g_rrg_dpo_full}"
RADEVAL_ENV="${RADEVAL_ENV:-/data/conda_envs/radeval311}"

cd "${ROOT_DIR}" || exit 1

run_cmd() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[START][${name}] $(date)" | tee -a "${logfile}"
  "$@" >>"${logfile}" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then
    echo "[DONE ][${name}] $(date)" | tee -a "${logfile}"
  else
    echo "[FAIL ][${name}] rc=${rc} $(date)" | tee -a "${logfile}"
  fi
  return ${rc}
}

run_py() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "$@"
}

check_path_or_warn() {
  local p="$1"
  if [ ! -e "${p}" ]; then
    echo "[WARN] Missing path: ${p}"
  fi
}

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] gpus=${GPU0},${GPU1}"

check_path_or_warn "${LLM_PATH}"
check_path_or_warn "${CT_RATE_RAW_MANIFEST}"
check_path_or_warn "${REX_MINI_MANIFEST}"
check_path_or_warn "${REX_100G_MANIFEST}"

# ---------------------------
# Stage 0: CPU prep for the largest dataset (CT-RATE)
# ---------------------------
PREP_LOG="${LOG_DIR}/prep_ct_rate_${TIMESTAMP}.log"

if [ ! -f "${CT_RATE_RRG_MANIFEST}" ]; then
  run_cmd "E0226_preprocess_rrg_dpo" "${PREP_LOG}" \
    "${PYTHON_BIN}" scripts/data/preprocess_manifest_rrg_dpo.py \
    --in-manifest "${CT_RATE_RAW_MANIFEST}" \
    --out-root "${CT_RATE_RRG_ROOT}" \
    --splits train val test \
    --dtype float16
else
  echo "[SKIP] CT-RATE RRG manifest exists: ${CT_RATE_RRG_MANIFEST}" | tee -a "${PREP_LOG}"
fi

if [ ! -f "${CT_RATE_CT2REP_ROOT}/reports.xlsx" ]; then
  run_cmd "E0227_build_ct2rep_dataset" "${PREP_LOG}" \
    "${PYTHON_BIN}" scripts/external/build_ct2rep_dataset_from_manifest.py \
    --manifest "${CT_RATE_RRG_MANIFEST}" \
    --out-root "${CT_RATE_CT2REP_ROOT}" \
    --splits train val test
else
  echo "[SKIP] CT2Rep dataset exists: ${CT_RATE_CT2REP_ROOT}/reports.xlsx" | tee -a "${PREP_LOG}"
fi

# ---------------------------
# Stage 1: Dual-GPU lanes
# ---------------------------
LANE0_LOG="${LOG_DIR}/lane_gpu${GPU0}_${TIMESTAMP}.log"
LANE1_LOG="${LOG_DIR}/lane_gpu${GPU1}_${TIMESTAMP}.log"

lane_gpu0() {
  # Highest priority: largest dataset + LLM full run.
  run_cmd "CT_RATE_LLM_full_methods_all" "${LANE0_LOG}" \
    run_py "${GPU0}" -m provetok.experiments.run_baselines \
    --dataset-type manifest \
    --manifest "${CT_RATE_TEST_MANIFEST}" \
    --split test \
    --resize-shape 128 128 128 \
    --pcg "${PCG_BACKEND}" \
    --llama2-path "${LLM_PATH}" \
    --llama2-quant fp16 \
    --n-samples 200 \
    --dump-text-pairs-jsonl ../pairs_llama2.jsonl \
    --output-dir ./outputs/E0301-ct_rate_llama2_full \
    --resume

  # Optional RadGenome full pass (if manifest exists)
  if [ -f "${RADGENOME_MANIFEST}" ]; then
    run_cmd "RADGENOME_LLM_full_methods_all" "${LANE0_LOG}" \
      run_py "${GPU0}" -m provetok.experiments.run_baselines \
      --dataset-type manifest \
      --manifest "${RADGENOME_MANIFEST}" \
      --split test \
      --resize-shape 64 64 64 \
      --pcg "${PCG_BACKEND}" \
      --llama2-path "${LLM_PATH}" \
      --llama2-quant fp16 \
      --n-samples 200 \
      --no-text-metrics \
      --output-dir ./outputs/E0304-radgenome_llama2_full \
      --resume
  else
    echo "[SKIP] RADGENOME manifest missing: ${RADGENOME_MANIFEST}" | tee -a "${LANE0_LOG}"
  fi
}

lane_gpu1() {
  # CT2Rep train/infer for CT-RATE in parallel with lane 0.
  run_cmd "E0228_ct2rep_train_full" "${LANE1_LOG}" \
    run_py "${GPU1}" scripts/external/train_ct2rep_baseline.py \
    --xlsxfile "${CT_RATE_CT2REP_ROOT}/reports.xlsx" \
    --trainfolder "${CT_RATE_CT2REP_ROOT}/train" \
    --validfolder "${CT_RATE_CT2REP_ROOT}/valid" \
    --save-dir "${CT2REP_RUN_DIR}" \
    --epochs 20 \
    --batch-size 1 \
    --num-workers 4 \
    --n-gpu 1 \
    --save-period 5 \
    --early-stop 5

  run_cmd "E0229_ct2rep_infer_full" "${LANE1_LOG}" \
    run_py "${GPU1}" scripts/external/infer_ct2rep_to_pred_jsonl.py \
    --train-args-json "${CT2REP_RUN_DIR}/ct2rep_train_args.json" \
    --xlsxfile "${CT_RATE_CT2REP_ROOT}/reports.xlsx" \
    --data-folder "${CT_RATE_CT2REP_ROOT}/test" \
    --ckpt "${CT2REP_RUN_DIR}/model_best.pth" \
    --out-jsonl outputs/E0229-ct2rep_pred_full/preds_ct2rep.jsonl \
    --method ct2rep \
    --batch-size 1 \
    --num-workers 4 \
    --device cuda

  # ReXGroundingCT mini full
  run_cmd "REX_MINI_LLM_full_methods_all" "${LANE1_LOG}" \
    run_py "${GPU1}" -m provetok.experiments.run_baselines \
    --dataset-type manifest \
    --manifest "${REX_MINI_MANIFEST}" \
    --split test \
    --resize-shape 64 64 64 \
    --pcg "${PCG_BACKEND}" \
    --llama2-path "${LLM_PATH}" \
    --llama2-quant fp16 \
    --n-samples 57 \
    --output-dir ./outputs/E0302-rexmini_llama2_full \
    --resume

  # ReXGroundingCT 100g full
  run_cmd "REX_100G_LLM_full_methods_all" "${LANE1_LOG}" \
    run_py "${GPU1}" -m provetok.experiments.run_baselines \
    --dataset-type manifest \
    --manifest "${REX_100G_MANIFEST}" \
    --split test \
    --resize-shape 64 64 64 \
    --pcg "${PCG_BACKEND}" \
    --llama2-path "${LLM_PATH}" \
    --llama2-quant fp16 \
    --n-samples 231 \
    --no-text-metrics \
    --output-dir ./outputs/E0303-rex100g_llama2_full \
    --resume
}

lane_gpu0 &
PID0=$!
lane_gpu1 &
PID1=$!

echo "[INFO] lane pids: gpu${GPU0}=${PID0}, gpu${GPU1}=${PID1}"
wait "${PID0}" || true
wait "${PID1}" || true

# ---------------------------
# Stage 2: Table1 merge (after both lanes)
# ---------------------------
POST_LOG="${LOG_DIR}/post_${TIMESTAMP}.log"

BASE_PAIRS="outputs/E0301-ct_rate_llama2_full/pairs_llama2.jsonl"
if [ ! -f "${BASE_PAIRS}" ]; then
  # run_baselines writes dump path relative to output dir (../pairs_llama2.jsonl)
  BASE_PAIRS="outputs/pairs_llama2.jsonl"
fi

TABLE_CMD=(
  "${PYTHON_BIN}" scripts/paper/run_table1_ct_rate.py
  --manifest "${CT_RATE_TEST_MANIFEST}"
  --split test
  --base-pairs-jsonl "${BASE_PAIRS}"
  --external-pred-jsonl outputs/E0229-ct2rep_pred_full/preds_ct2rep.jsonl
  --run-proof-external
  --out-dir outputs/E0305-table1_all_dataset_round1
  --n-bootstrap 10000
  --holm-family all
)

if [ "${RUN_RADEVAL}" = "1" ]; then
  TABLE_CMD+=(--run-radeval --radeval-env "${RADEVAL_ENV}")
fi

run_cmd "E0305_table1_merge" "${POST_LOG}" "${TABLE_CMD[@]}"

echo "[INFO] all done. logs: ${LOG_DIR}"
