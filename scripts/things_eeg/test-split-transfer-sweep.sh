#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2099}"

OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-test_split_transfer_sweep_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

CALIBRATION_SIZE="${CALIBRATION_SIZE:-100}"
EVAL_SIZE="${EVAL_SIZE:-100}"

SAW_SHRINK="${SAW_SHRINK:-0.85}"
CSLS_K="${CSLS_K:-1}"
SINKHORN_TAU="${SINKHORN_TAU:-0.1}"
SINKHORN_ITERS="${SINKHORN_ITERS:-30}"
SOFT_STEPS="${SOFT_STEPS:-10}"
SOFT_POWER="${SOFT_POWER:-1.0}"

FROZEN_ALPHA_VALUES="${FROZEN_ALPHA_VALUES:-0.4 0.45 0.5 0.55 0.6}"
FROZEN_TAU_VALUES="${FROZEN_TAU_VALUES:-0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4}"

SAW_DIAG="${SAW_DIAG:-0}"
SAW_NO_RENORM="${SAW_NO_RENORM:-0}"
SOFT_NORM_INPUTS="${SOFT_NORM_INPUTS:-0}"

mkdir -p "$RUN_ROOT"

read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"
read -r -a FROZEN_ALPHA_ARR <<< "$FROZEN_ALPHA_VALUES"
read -r -a FROZEN_TAU_ARR <<< "$FROZEN_TAU_VALUES"

ARGS=(
    --source_run_dir "$SOURCE_RUN_DIR"
    --output_dir "$RUN_ROOT"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --calibration_size "$CALIBRATION_SIZE"
    --eval_size "$EVAL_SIZE"
    --sattc_saw_shrink "$SAW_SHRINK"
    --sattc_csls_k "$CSLS_K"
    --sattc_sinkhorn_tau "$SINKHORN_TAU"
    --sattc_sinkhorn_iters "$SINKHORN_ITERS"
    --sattc_soft_procrustes_steps "$SOFT_STEPS"
    --sattc_soft_procrustes_power "$SOFT_POWER"
    --held_out_subjects "${HELD_OUT_SUBJECT_ARR[@]}"
    --frozen_alpha_values "${FROZEN_ALPHA_ARR[@]}"
    --frozen_tau_values "${FROZEN_TAU_ARR[@]}"
)

if [ "$SAW_DIAG" = "1" ]; then
    ARGS+=(--sattc_saw_diag)
fi
if [ "$SAW_NO_RENORM" = "1" ]; then
    ARGS+=(--sattc_saw_no_renorm)
fi
if [ "$SOFT_NORM_INPUTS" = "1" ]; then
    ARGS+=(--sattc_soft_procrustes_normalize_inputs)
fi

echo "=========================================================="
echo "Test-split transfer sweep"
echo "source_run_dir=${SOURCE_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "subjects=${HELD_OUT_SUBJECTS}"
echo "split=${CALIBRATION_SIZE}/${EVAL_SIZE}"
echo "saw=${SAW_SHRINK} csls_k=${CSLS_K} tau=${SINKHORN_TAU} iters=${SINKHORN_ITERS} steps=${SOFT_STEPS} power=${SOFT_POWER}"
echo "frozen_alpha_values=${FROZEN_ALPHA_VALUES}"
echo "frozen_tau_values=${FROZEN_TAU_VALUES}"
echo "=========================================================="

python3 "${REPO_ROOT}/scripts/things_eeg/transfer_calibration_experiment.py" "${ARGS[@]}"

echo "=========================================================="
echo "Transfer sweep completed."
echo "All results: ${RUN_ROOT}/all_subject_results.csv"
echo "Average results: ${RUN_ROOT}/average_results.csv"
echo "Config: ${RUN_ROOT}/experiment_config.json"
echo "=========================================================="
