#!/bin/bash
set -euo pipefail
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-3300}"
PROGRESSIVE_REPEATS="${PROGRESSIVE_REPEATS:-10}"

OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-inductive_tta_transfer_suite_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.94}"
BASE_CSLS_K="${BASE_CSLS_K:-3}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"

ALPHA_VALUES="${ALPHA_VALUES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
FIT_SIZES="${FIT_SIZES:-$(seq 5 5 200 | tr '\n' ' ')}"

mkdir -p "$RUN_ROOT"

read -r -a SUBJECT_ARR <<< "$SUBJECTS"
read -r -a ALPHA_ARR <<< "$ALPHA_VALUES"
read -r -a FIT_SIZE_ARR <<< "$FIT_SIZES"

ARGS=(
    --source_run_dir "$SOURCE_RUN_DIR"
    --output_dir "$RUN_ROOT"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --progressive_repeats "$PROGRESSIVE_REPEATS"
    --subjects "${SUBJECT_ARR[@]}"
    --alpha_values "${ALPHA_ARR[@]}"
    --fit_sizes "${FIT_SIZE_ARR[@]}"
    --sattc_saw_shrink "$BASE_SAW_SHRINK"
    --sattc_csls_k "$BASE_CSLS_K"
    --sattc_sinkhorn_tau "$BASE_SINKHORN_TAU"
    --sattc_sinkhorn_iters "$BASE_SINKHORN_ITERS"
    --sattc_soft_procrustes_steps "$BASE_SOFT_STEPS"
    --sattc_soft_procrustes_power "$BASE_SOFT_POWER"
)

echo "=========================================================="
echo "Inductive-style SATTC transfer suite"
echo "source_run_dir=${SOURCE_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "subjects=${SUBJECTS}"
echo "fit_sizes=${FIT_SIZES}"
echo "alpha_values=${ALPHA_VALUES}"
echo "progressive_repeats=${PROGRESSIVE_REPEATS}"
echo "params: saw=${BASE_SAW_SHRINK} csls_k=${BASE_CSLS_K} tau=${BASE_SINKHORN_TAU} iters=${BASE_SINKHORN_ITERS} steps=${BASE_SOFT_STEPS} power=${BASE_SOFT_POWER}"
echo "=========================================================="

python3 "${REPO_ROOT}/scripts/things_eeg/inductive_tta_transfer_suite.py" "${ARGS[@]}"

echo "=========================================================="
echo "Completed."
echo "Baselines: ${RUN_ROOT}/baseline_average_results.csv"
echo "Progressive averages: ${RUN_ROOT}/progressive_average_results.csv"
echo "Best alpha curve: ${RUN_ROOT}/progressive_best_alpha_results.csv"
echo "Overview: ${RUN_ROOT}/overview_summary.csv"
echo "=========================================================="
