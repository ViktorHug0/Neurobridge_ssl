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
SPLIT_SEED="${SPLIT_SEED:-2099}"
SPLIT_A_SIZE="${SPLIT_A_SIZE:-100}"
SPLIT_B_SIZE="${SPLIT_B_SIZE:-100}"

DEFAULT_SAW_SHRINK="${DEFAULT_SAW_SHRINK:-0.85}"

TRANSFER_SAW_SHRINK="${TRANSFER_SAW_SHRINK:-0.85}"
TRANSFER_CSLS_K="${TRANSFER_CSLS_K:-1}"
TRANSFER_SINKHORN_TAU="${TRANSFER_SINKHORN_TAU:-0.1}"
TRANSFER_SINKHORN_ITERS="${TRANSFER_SINKHORN_ITERS:-20}"
TRANSFER_SOFT_STEPS="${TRANSFER_SOFT_STEPS:-10}"
TRANSFER_SOFT_POWER="${TRANSFER_SOFT_POWER:-1.0}"
TRANSFER_NO_RENORM="${TRANSFER_NO_RENORM:-1}"
TRANSFER_MATCH_EVAL_SAW="${TRANSFER_MATCH_EVAL_SAW:-1}"

ALPHA_VALUES="${ALPHA_VALUES:-0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128}"

OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-source_run_split_transfer_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"

mkdir -p "$RUN_ROOT"
read -r -a ALPHA_ARR <<< "$ALPHA_VALUES"

echo "=========================================================="
echo "Source-run split-transfer sweep"
echo "source_run_dir=${SOURCE_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "split_seed=${SPLIT_SEED}"
echo "split_a_size=${SPLIT_A_SIZE:-default_half}"
echo "split_b_size=${SPLIT_B_SIZE:-remaining_half}"
echo "default_saw_shrink=${DEFAULT_SAW_SHRINK}"
if [ "$TRANSFER_MATCH_EVAL_SAW" = "1" ]; then
    echo "transfer=match_eval_saw_k${TRANSFER_CSLS_K}_tau${TRANSFER_SINKHORN_TAU}_steps${TRANSFER_SOFT_STEPS}_pow${TRANSFER_SOFT_POWER}_iters${TRANSFER_SINKHORN_ITERS}"
else
    echo "transfer=saw${TRANSFER_SAW_SHRINK}_k${TRANSFER_CSLS_K}_tau${TRANSFER_SINKHORN_TAU}_steps${TRANSFER_SOFT_STEPS}_pow${TRANSFER_SOFT_POWER}_iters${TRANSFER_SINKHORN_ITERS}_norenorm${TRANSFER_NO_RENORM}"
fi
echo "alpha_values=${ALPHA_VALUES}"
echo "=========================================================="

ARGS=(
    --source_run_dir "$SOURCE_RUN_DIR"
    --output_dir "$RUN_ROOT"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --split_seed "$SPLIT_SEED"
    --default_saw_shrink "$DEFAULT_SAW_SHRINK"
    --transfer_saw_shrink "$TRANSFER_SAW_SHRINK"
    --transfer_csls_k "$TRANSFER_CSLS_K"
    --transfer_sinkhorn_tau "$TRANSFER_SINKHORN_TAU"
    --transfer_sinkhorn_iters "$TRANSFER_SINKHORN_ITERS"
    --transfer_soft_procrustes_steps "$TRANSFER_SOFT_STEPS"
    --transfer_soft_procrustes_power "$TRANSFER_SOFT_POWER"
    --alpha_values "${ALPHA_ARR[@]}"
)

if [ -n "$SPLIT_A_SIZE" ]; then
    ARGS+=(--split_a_size "$SPLIT_A_SIZE")
fi
if [ -n "$SPLIT_B_SIZE" ]; then
    ARGS+=(--split_b_size "$SPLIT_B_SIZE")
fi

if [ "$TRANSFER_NO_RENORM" = "1" ]; then
    ARGS+=(--transfer_no_renorm)
else
    ARGS+=(--transfer_renorm)
fi

if [ "$TRANSFER_MATCH_EVAL_SAW" = "1" ]; then
    ARGS+=(--transfer_match_eval_saw)
else
    ARGS+=(--transfer_use_fixed_saw)
fi

python3 "${REPO_ROOT}/scripts/things_eeg/session_split_transfer_experiment.py" "${ARGS[@]}"

echo "=========================================================="
echo "Source-run split-transfer sweep completed."
echo "Per-subject results: ${RUN_ROOT}/all_subject_results.csv"
echo "Run averages: ${RUN_ROOT}/run_average_results.csv"
echo "Split manifest: ${RUN_ROOT}/shared_split_manifest.json"
echo "Config: ${RUN_ROOT}/experiment_config.json"
echo "=========================================================="
