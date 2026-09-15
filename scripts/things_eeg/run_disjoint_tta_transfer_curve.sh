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
NUM_REPEATS="${NUM_REPEATS:-5}"

OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-}"
RUN_ROOT="${RUN_ROOT:-}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
FIT_SIZES="${FIT_SIZES:-$(seq 5 5 195 | tr '\n' ' ')}"
ALPHA_VALUES="${ALPHA_VALUES:-0.0 0.25 0.5 0.75 1.0}"

BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.94}"
BASE_CSLS_K="${BASE_CSLS_K:-3}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"

if [ -z "$RUN_ROOT" ]; then
    if [ -n "$RUN_TAG" ]; then
        RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
    else
        shopt -s nullglob
        MATCHING_RUNS=("${OUTPUT_ROOT}"/disjoint_tta_transfer_curve_*)
        shopt -u nullglob

        LATEST_INCOMPLETE_RUN=""
        if [ ${#MATCHING_RUNS[@]} -gt 0 ]; then
            IFS=$'\n' MATCHING_RUNS=($(printf '%s\n' "${MATCHING_RUNS[@]}" | sort))
            unset IFS
            for ((idx=${#MATCHING_RUNS[@]}-1; idx>=0; idx--)); do
                CANDIDATE_RUN="${MATCHING_RUNS[$idx]}"
                if [ -f "${CANDIDATE_RUN}/disjoint_transfer_raw_results.csv" ] && [ ! -f "${CANDIDATE_RUN}/disjoint_transfer_aggregate_results.csv" ]; then
                    LATEST_INCOMPLETE_RUN="${CANDIDATE_RUN}"
                    break
                fi
            done
        fi

        if [ -n "$LATEST_INCOMPLETE_RUN" ]; then
            RUN_ROOT="$LATEST_INCOMPLETE_RUN"
            echo "Resuming previous incomplete run: ${RUN_ROOT}"
        else
            RUN_TAG="disjoint_tta_transfer_curve_$(date +'%Y%m%d-%H%M%S')"
            RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
        fi
    fi
fi

mkdir -p "$RUN_ROOT"

read -r -a SUBJECT_ARR <<< "$SUBJECTS"
read -r -a FIT_SIZE_ARR <<< "$FIT_SIZES"
read -r -a ALPHA_ARR <<< "$ALPHA_VALUES"

ARGS=(
    --source_run_dir "$SOURCE_RUN_DIR"
    --output_dir "$RUN_ROOT"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --num_repeats "$NUM_REPEATS"
    --subjects "${SUBJECT_ARR[@]}"
    --fit_sizes "${FIT_SIZE_ARR[@]}"
    --alpha_values "${ALPHA_ARR[@]}"
    --sattc_saw_shrink "$BASE_SAW_SHRINK"
    --sattc_csls_k "$BASE_CSLS_K"
    --sattc_sinkhorn_tau "$BASE_SINKHORN_TAU"
    --sattc_sinkhorn_iters "$BASE_SINKHORN_ITERS"
    --sattc_soft_procrustes_steps "$BASE_SOFT_STEPS"
    --sattc_soft_procrustes_power "$BASE_SOFT_POWER"
)

echo "=========================================================="
echo "Disjoint TTA transfer curve experiment"
echo "source_run_dir=${SOURCE_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "subjects=${SUBJECTS}"
echo "fit_sizes=${FIT_SIZES}"
echo "alpha_values=${ALPHA_VALUES}"
echo "num_repeats=${NUM_REPEATS}"
echo "params: saw=${BASE_SAW_SHRINK} csls_k=${BASE_CSLS_K} tau=${BASE_SINKHORN_TAU} iters=${BASE_SINKHORN_ITERS} steps=${BASE_SOFT_STEPS} power=${BASE_SOFT_POWER}"
echo "=========================================================="

python3 "${REPO_ROOT}/scripts/things_eeg/disjoint_tta_transfer_curve.py" "${ARGS[@]}"

echo "=========================================================="
echo "Completed."
echo "Raw results: ${RUN_ROOT}/disjoint_transfer_raw_results.csv"
echo "Aggregate results: ${RUN_ROOT}/disjoint_transfer_aggregate_results.csv"
echo "Summary: ${RUN_ROOT}/disjoint_transfer_summary.csv"
echo "Plots: ${RUN_ROOT}/*gain_plot.png"
echo "=========================================================="
