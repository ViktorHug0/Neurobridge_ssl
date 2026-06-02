#!/bin/bash
set -euo pipefail
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/things_eeg/tta_rebuttal}"
RUN_TAG="${RUN_TAG:-rebuttal_suite_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"

BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.94}"
BASE_CSLS_K="${BASE_CSLS_K:-3}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"

RUN_PROGRESSIVE="${RUN_PROGRESSIVE:-false}"
RUN_SPLIT_TRANSFER="${RUN_SPLIT_TRANSFER:-false}"
RUN_REPETITION_ABLATION="${RUN_REPETITION_ABLATION:-false}"
RUN_FEWSHOT="${RUN_FEWSHOT:-true}"
RUN_TRAINSET_TRANSFER="${RUN_TRAINSET_TRANSFER:-false}"

mkdir -p "$RUN_ROOT"
read -r -a SUBJECT_ARR <<< "$SUBJECTS"

COMMON_ARGS=(
    --source_run_dir "$SOURCE_RUN_DIR"
    --subjects "${SUBJECT_ARR[@]}"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --sattc_saw_shrink "$BASE_SAW_SHRINK"
    --sattc_csls_k "$BASE_CSLS_K"
    --sattc_sinkhorn_tau "$BASE_SINKHORN_TAU"
    --sattc_sinkhorn_iters "$BASE_SINKHORN_ITERS"
    --sattc_soft_procrustes_steps "$BASE_SOFT_STEPS"
    --sattc_soft_procrustes_power "$BASE_SOFT_POWER"
)

echo "=========================================================="
echo "TTA rebuttal suite"
echo "source_run_dir=${SOURCE_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "subjects=${SUBJECTS}"
echo "params: saw=${BASE_SAW_SHRINK} csls_k=${BASE_CSLS_K} tau=${BASE_SINKHORN_TAU} iters=${BASE_SINKHORN_ITERS} steps=${BASE_SOFT_STEPS} power=${BASE_SOFT_POWER}"
echo "=========================================================="

if [ "$RUN_PROGRESSIVE" = "true" ]; then
    python3 "${SCRIPT_DIR}/run_progressive_calibration.py" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${RUN_ROOT}/progressive_calibration"
fi

if [ "$RUN_SPLIT_TRANSFER" = "true" ]; then
    python3 "${SCRIPT_DIR}/run_split_transfer.py" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${RUN_ROOT}/split_transfer"
fi

if [ "$RUN_REPETITION_ABLATION" = "true" ]; then
    python3 "${SCRIPT_DIR}/run_repetition_ablation.py" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${RUN_ROOT}/repetition_ablation" \
        --repetition_counts 80 70 60 50 40 30 20 10
fi

if [ "$RUN_FEWSHOT" = "true" ]; then
    python3 "${SCRIPT_DIR}/run_fewshot_subject_adaptation.py" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${RUN_ROOT}/fewshot_subject_adaptation" \
        --train_size "${FEWSHOT_TRAIN_SIZE:-80}" \
        --val_size "${FEWSHOT_VAL_SIZE:-20}"
fi

if [ "$RUN_TRAINSET_TRANSFER" = "true" ]; then
    read -r -a TRAINSET_SIZE_ARR <<< "${TRAINSET_CALIBRATION_SIZES:-100 500 1000 5000 10000 all}"
    python3 "${SCRIPT_DIR}/run_trainset_rotation_transfer.py" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${RUN_ROOT}/trainset_rotation_transfer" \
        --calibration_sizes "${TRAINSET_SIZE_ARR[@]}"
fi

echo "=========================================================="
echo "Completed TTA rebuttal suite."
echo "Results: ${RUN_ROOT}"
echo "=========================================================="
