#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-orth_adapt_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
SEED="${SEED:-2099}"

FEATURE_DIM="${FEATURE_DIM:-128}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-128}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4}"

EVAL_MODE="${EVAL_MODE:-saw}"
EVAL_SAW_SHRINK="${EVAL_SAW_SHRINK:-0.85}"
EVAL_CSLS_K="${EVAL_CSLS_K:-1}"

SUBJECT_ADAPT_LAMBDA="${SUBJECT_ADAPT_LAMBDA:-1.0}"
SUBJECT_ADAPT_SPLIT_A_RATIO="${SUBJECT_ADAPT_SPLIT_A_RATIO:-0.5}"
SUBJECT_ADAPT_MIN_SAMPLES="${SUBJECT_ADAPT_MIN_SAMPLES:-8}"
SUBJECT_ADAPT_SAW_SHRINK="${SUBJECT_ADAPT_SAW_SHRINK:-0.85}"
SUBJECT_ADAPT_CSLS_K="${SUBJECT_ADAPT_CSLS_K:-1}"
SUBJECT_ADAPT_SINKHORN_TAU="${SUBJECT_ADAPT_SINKHORN_TAU:-0.1}"
SUBJECT_ADAPT_SINKHORN_ITERS="${SUBJECT_ADAPT_SINKHORN_ITERS:-20}"
SUBJECT_ADAPT_SOFT_STEPS="${SUBJECT_ADAPT_SOFT_STEPS:-10}"
SUBJECT_ADAPT_SOFT_POWER="${SUBJECT_ADAPT_SOFT_POWER:-1.0}"
SUBJECT_ADAPT_NO_RENORM="${SUBJECT_ADAPT_NO_RENORM:-1}"

mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

echo "=========================================================="
echo "Inter-subject orthogonal adaptation"
echo "run_root=${RUN_ROOT}"
echo "subjects=${HELD_OUT_SUBJECTS}"
echo "feature_dim=${FEATURE_DIM} eeg_backbone_dim=${EEG_BACKBONE_DIM}"
echo "subject_adapt=lambda${SUBJECT_ADAPT_LAMBDA}_split${SUBJECT_ADAPT_SPLIT_A_RATIO}_saw${SUBJECT_ADAPT_SAW_SHRINK}_k${SUBJECT_ADAPT_CSLS_K}_tau${SUBJECT_ADAPT_SINKHORN_TAU}_steps${SUBJECT_ADAPT_SOFT_STEPS}_pow${SUBJECT_ADAPT_SOFT_POWER}_iters${SUBJECT_ADAPT_SINKHORN_ITERS}_norenorm${SUBJECT_ADAPT_NO_RENORM}"
echo "=========================================================="

for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
do
    OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
    echo "Training subject ${SUB_ID}..."

    TRAIN_IDS=""
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        if [ "$i" -ne "$SUB_ID" ]; then
            TRAIN_IDS+="$i "
        fi
    done

    ARGS=(
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --learning_rate "$LEARNING_RATE"
        --output_name "$OUTPUT_NAME"
        --eeg_encoder_type "$EEG_ENCODER_TYPE"
        --train_subject_ids $TRAIN_IDS
        --test_subject_ids "$SUB_ID"
        --softplus
        --num_epochs "$NUM_EPOCHS"
        --image_feature_dir "$IMAGE_FEATURE_DIR"
        --text_feature_dir "$TEXT_FEATURE_DIR"
        --eeg_data_dir "$EEG_DATA_DIR"
        --device "$DEVICE"
        --output_dir "$RUN_ROOT"
        --img_l2norm
        --projector "$PROJECTOR"
        --data_average
        --save_weights
        --feature_dim "$FEATURE_DIM"
        --eeg_backbone_dim "$EEG_BACKBONE_DIM"
        --ssl_lambda 0
        --multi_positive_loss
        --grouped_batch_sampler
        --samples_per_image 6
        --eval_mode "$EVAL_MODE"
        --sattc_saw_shrink "$EVAL_SAW_SHRINK"
        --sattc_csls_k "$EVAL_CSLS_K"
        --subject_adapt_lambda "$SUBJECT_ADAPT_LAMBDA"
        --subject_adapt_split_a_ratio "$SUBJECT_ADAPT_SPLIT_A_RATIO"
        --subject_adapt_min_samples_per_subject "$SUBJECT_ADAPT_MIN_SAMPLES"
        --subject_adapt_saw_shrink "$SUBJECT_ADAPT_SAW_SHRINK"
        --subject_adapt_csls_k "$SUBJECT_ADAPT_CSLS_K"
        --subject_adapt_sinkhorn_tau "$SUBJECT_ADAPT_SINKHORN_TAU"
        --subject_adapt_sinkhorn_iters "$SUBJECT_ADAPT_SINKHORN_ITERS"
        --subject_adapt_soft_procrustes_steps "$SUBJECT_ADAPT_SOFT_STEPS"
        --subject_adapt_soft_procrustes_power "$SUBJECT_ADAPT_SOFT_POWER"
        --seed "$SEED"
    )

    if [ "$SUBJECT_ADAPT_NO_RENORM" = "1" ]; then
        ARGS+=(--subject_adapt_saw_no_renorm)
    fi

    python3 "${REPO_ROOT}/train.py" "${ARGS[@]}"
done

python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$RUN_ROOT" --output_name "inter_subject_summary.csv"

echo "=========================================================="
echo "Orthogonal adaptation run completed."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${RUN_ROOT}/inter_subject_summary.csv"
echo "=========================================================="
