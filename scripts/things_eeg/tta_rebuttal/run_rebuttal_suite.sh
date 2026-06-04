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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/things_eeg/tta_rebuttal}"
if [ -n "${RUN_ROOT:-}" ]; then
    RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
else
    RUN_TAG="${RUN_TAG:-rebuttal_suite_$(date +'%Y%m%d-%H%M%S')}"
    RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
fi

# Base models: 64-dim mixup (new mainline) then 512-dim TSConv (rebuttal_suite_20260602-134420).
SOURCE_RUN_DIR_FEATDIM64="${SOURCE_RUN_DIR_FEATDIM64:-${REPO_ROOT}/results/things_eeg/inter-subjects/mixup_20260421-190931/mix_raw_eeg_pairwise_linear_a0p5_seed3300}"
SOURCE_RUN_DIR_FEATDIM512="${SOURCE_RUN_DIR_FEATDIM512:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"

# RUN_BOTH_MODELS=true (default): run featdim64 then featdim512 under RUN_ROOT/{model_tag}/.
# RUN_BOTH_MODELS=false: single model via SOURCE_RUN_DIR (optional MODEL_TAG subfolder).
RUN_BOTH_MODELS="${RUN_BOTH_MODELS:-true}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
MODEL_TAG="${MODEL_TAG:-}"

BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.94}"
BASE_CSLS_K="${BASE_CSLS_K:-3}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"

RUN_PROGRESSIVE="${RUN_PROGRESSIVE:-false}"
RUN_SPLIT_TRANSFER="${RUN_SPLIT_TRANSFER:-false}"
RUN_REPETITION_ABLATION="${RUN_REPETITION_ABLATION:-false}"
RUN_FEWSHOT="${RUN_FEWSHOT:-false}"
RUN_TRAINSET_TRANSFER="${RUN_TRAINSET_TRANSFER:-false}"
RUN_SUBSPACE_ROTATION="${RUN_SUBSPACE_ROTATION:-false}"
RUN_INDUCTIVE_ROTATION="${RUN_INDUCTIVE_ROTATION:-true}"
RUN_FEWSHOT_FINETUNING="${RUN_FEWSHOT_FINETUNING:-false}"

mkdir -p "$RUN_ROOT"
read -r -a SUBJECT_ARR <<< "$SUBJECTS"

run_experiments_for_model() {
    local source_run_dir="$1"
    local model_tag="$2"
    local model_root="${RUN_ROOT}"
    if [ -n "$model_tag" ]; then
        model_root="${RUN_ROOT}/${model_tag}"
    fi

    if [ ! -d "$source_run_dir" ]; then
        echo "ERROR: source_run_dir does not exist: ${source_run_dir}" >&2
        return 1
    fi

    local -a common_args=(
        --source_run_dir "$source_run_dir"
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

    echo "----------------------------------------------------------"
    echo "Model pass: ${model_tag:-single}"
    echo "source_run_dir=${source_run_dir}"
    echo "output_root=${model_root}"
    echo "----------------------------------------------------------"

    if [ "$RUN_PROGRESSIVE" = "true" ]; then
        python3 "${SCRIPT_DIR}/run_progressive_calibration.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/progressive_calibration"
    fi

    if [ "$RUN_SPLIT_TRANSFER" = "true" ]; then
        python3 "${SCRIPT_DIR}/run_split_transfer.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/split_transfer"
    fi

    if [ "$RUN_REPETITION_ABLATION" = "true" ]; then
        python3 "${SCRIPT_DIR}/run_repetition_ablation.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/repetition_ablation" \
            --repetition_counts 80 70 60 50 40 30 20 10
    fi

    if [ "$RUN_FEWSHOT" = "true" ]; then
        python3 "${SCRIPT_DIR}/run_fewshot_subject_adaptation.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/fewshot_subject_adaptation" \
            --train_size "${FEWSHOT_TRAIN_SIZE:-80}" \
            --val_size "${FEWSHOT_VAL_SIZE:-20}"
    fi

    if [ "$RUN_TRAINSET_TRANSFER" = "true" ]; then
        read -r -a trainset_size_arr <<< "${TRAINSET_CALIBRATION_SIZES:-5 10 20 50 100 500 1000 all}"
        local trainset_out="${model_root}/trainset_rotation_transfer"
        if [ "$model_tag" = "featdim64_mixup" ]; then
            trainset_out="${model_root}/trainset_rotation_transfer_featdim64_mixup"
        fi
        python3 "${SCRIPT_DIR}/run_trainset_rotation_transfer.py" \
            "${common_args[@]}" \
            --output_dir "${trainset_out}" \
            --calibration_sizes "${trainset_size_arr[@]}"
    fi

    if [ "$RUN_SUBSPACE_ROTATION" = "true" ]; then
        read -r -a subspace_size_arr <<< "${SUBSPACE_CALIBRATION_SIZES:-5 10 20 50 100 500 1000 all}"
        read -r -a subspace_dims_arr <<< "${SUBSPACE_DIMS:-1 3 5}"
        python3 "${SCRIPT_DIR}/run_subspace_rotation.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/subspace_rotation" \
            --calibration_sizes "${subspace_size_arr[@]}" \
            --subspace_dims "${subspace_dims_arr[@]}"
    fi

    if [ "$RUN_INDUCTIVE_ROTATION" = "true" ]; then
        read -r -a inductive_size_arr <<< "${INDUCTIVE_CALIBRATION_SIZES:-5 10 20 50 100 200 500 1000 2000 5000 10000 all}"
        read -r -a inductive_alpha_arr <<< "${INDUCTIVE_ALPHA_VALUES:-0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0}"
        read -r -a inductive_seed_arr <<< "${INDUCTIVE_SEEDS:-3300}"
        python3 "${SCRIPT_DIR}/run_inductive_rotation_sweep.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/inductive_rotation_sweep" \
            --calibration_sizes "${inductive_size_arr[@]}" \
            --alpha_values "${inductive_alpha_arr[@]}" \
            --seeds "${inductive_seed_arr[@]}"
    fi

    if [ "$RUN_FEWSHOT_FINETUNING" = "true" ]; then
        read -r -a fewshot_ft_size_arr <<< "${FEWSHOT_FT_CALIBRATION_SIZES:-10 20 50 100 200 500 1000 2000 5000 all}"
        read -r -a fewshot_ft_seed_arr <<< "${FEWSHOT_FT_SEEDS:-3300 3301 3302}"
        python3 "${SCRIPT_DIR}/run_fewshot_finetuning.py" \
            "${common_args[@]}" \
            --output_dir "${model_root}/fewshot_finetuning" \
            --calibration_sizes "${fewshot_ft_size_arr[@]}" \
            --seeds "${fewshot_ft_seed_arr[@]}" \
            --lr_projector "${FEWSHOT_FT_LR_PROJ:-3e-4}" \
            --lr_full "${FEWSHOT_FT_LR_FULL:-1e-4}" \
            --adapter_lr "${FEWSHOT_FT_LR_ADAPTER:-1e-3}" \
            --adapter_rank "${FEWSHOT_FT_ADAPTER_RANK:-16}" \
            --max_epochs "${FEWSHOT_FT_MAX_EPOCHS:-200}" \
            --patience "${FEWSHOT_FT_PATIENCE:-20}" \
            --batch_size_ft "${FEWSHOT_FT_BATCH_SIZE:-256}" \
            --encode_batch_size "${FEWSHOT_FT_ENCODE_BATCH_SIZE:-256}"
    fi
}

echo "=========================================================="
echo "TTA rebuttal suite"
echo "run_root=${RUN_ROOT}"
echo "subjects=${SUBJECTS}"
echo "run_both_models=${RUN_BOTH_MODELS}"
echo "params: saw=${BASE_SAW_SHRINK} csls_k=${BASE_CSLS_K} tau=${BASE_SINKHORN_TAU} iters=${BASE_SINKHORN_ITERS} steps=${BASE_SOFT_STEPS} power=${BASE_SOFT_POWER}"
echo "flags: progressive=${RUN_PROGRESSIVE} split=${RUN_SPLIT_TRANSFER} repetition=${RUN_REPETITION_ABLATION} fewshot=${RUN_FEWSHOT} trainset=${RUN_TRAINSET_TRANSFER} subspace=${RUN_SUBSPACE_ROTATION} inductive=${RUN_INDUCTIVE_ROTATION} fewshot_ft=${RUN_FEWSHOT_FINETUNING}"
echo "=========================================================="

if [ "$RUN_BOTH_MODELS" = "true" ]; then
    run_experiments_for_model "$SOURCE_RUN_DIR_FEATDIM64" "featdim64_mixup"
    run_experiments_for_model "$SOURCE_RUN_DIR_FEATDIM512" "featdim512"
else
    if [ -z "$SOURCE_RUN_DIR" ]; then
        if [ "$MODEL_TAG" = "featdim512" ]; then
            SOURCE_RUN_DIR="$SOURCE_RUN_DIR_FEATDIM512"
        elif [ "$MODEL_TAG" = "featdim64_mixup" ]; then
            SOURCE_RUN_DIR="$SOURCE_RUN_DIR_FEATDIM64"
        fi
    fi
    if [ -z "$SOURCE_RUN_DIR" ]; then
        echo "ERROR: RUN_BOTH_MODELS=false requires SOURCE_RUN_DIR or MODEL_TAG=featdim512|featdim64_mixup." >&2
        exit 1
    fi
    run_experiments_for_model "$SOURCE_RUN_DIR" "$MODEL_TAG"
fi

echo "=========================================================="
echo "Completed TTA rebuttal suite."
echo "Results: ${RUN_ROOT}"
if [ "$RUN_BOTH_MODELS" = "true" ]; then
    echo "  featdim64_mixup: ${RUN_ROOT}/featdim64_mixup"
    echo "  featdim512:      ${RUN_ROOT}/featdim512"
fi
echo "=========================================================="
