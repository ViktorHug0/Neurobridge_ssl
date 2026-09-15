#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi
cd "$REPO_ROOT"

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
SEED="${SEED:-2099}"
SUBJECT_POOL=(1 2 3 4 5)
ALL_SUBJECTS=(1 2 3 4 5 6 7 8 9 10)
MIXUP_ALPHA="${MIXUP_ALPHA:-0.5}"
REG_LAMBDAS="${REG_LAMBDAS:-0.01 0.05 0.1 0.5 1.0 5}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
SESSION_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
SESSION_DIR="${OUTPUT_DIR_BASE}/mixup_reg_sweep_${SESSION_TIMESTAMP}_seed${SEED}"
SESSION_SUMMARY="${SESSION_DIR}/mixup_reg_summary.csv"
SELECTED_CHANNELS=()

mkdir -p "$SESSION_DIR"
read -r -a REG_LAMBDA_ARR <<< "$REG_LAMBDAS"

upsert_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$run_dir" "$config_name" "$SESSION_SUMMARY" <<'PY'
import os
import pandas as pd
import sys

run_dir, config_name, session_summary_path = sys.argv[1:4]
inter_summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
if not os.path.exists(inter_summary_path):
    raise SystemExit(0)
df = pd.read_csv(inter_summary_path)
avg_row = df[df["sub"] == "Average"].copy()
if avg_row.empty:
    raise SystemExit(0)
avg_row.insert(0, "config", config_name)
if os.path.exists(session_summary_path):
    session_df = pd.read_csv(session_summary_path)
    session_df = session_df[session_df["config"] != config_name]
    avg_row = pd.concat([session_df, avg_row], ignore_index=True)
avg_row.to_csv(session_summary_path, index=False)
PY
}

CONFIG_NAMES=()
CONFIG_ARGS=()

for REG_LAMBDA in "${REG_LAMBDA_ARR[@]}"
do
    LAMBDA_TAG="${REG_LAMBDA//./p}"
    CONFIG_NAMES+=("mixup_reg_l${LAMBDA_TAG}")
    CONFIG_ARGS+=(
        "--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha ${MIXUP_ALPHA} --subject_mixup_reg_lambda ${REG_LAMBDA}"
    )
done

for c_idx in "${!CONFIG_NAMES[@]}"
do
    CONFIG_NAME="${CONFIG_NAMES[$c_idx]}"
    EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
    RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
    mkdir -p "$RUN_DIR"

    if [ -f "${RUN_DIR}/inter_subject_summary.csv" ]; then
        echo "Skipping completed config ${CONFIG_NAME}"
        upsert_average_row "$CONFIG_NAME" "$RUN_DIR"
        continue
    fi

    echo "=========================================================="
    echo "Running ${CONFIG_NAME}"
    echo "Args: ${EXTRA_ARGS}"
    echo "=========================================================="

    for SUB_ID in "${SUBJECT_POOL[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
            echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
            continue
        fi
        TRAIN_IDS=()
        for TRAIN_SUB_ID in "${ALL_SUBJECTS[@]}"
        do
            if [ "$TRAIN_SUB_ID" -ne "$SUB_ID" ]; then
                TRAIN_IDS+=("$TRAIN_SUB_ID")
            fi
        done

        python3 train.py \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --learning_rate "$LEARNING_RATE" \
            --output_name "$OUTPUT_NAME" \
            --eeg_encoder_type "$EEG_ENCODER_TYPE" \
            --train_subject_ids "${TRAIN_IDS[@]}" \
            --test_subject_ids "$SUB_ID" \
            --softplus \
            --num_epochs "$NUM_EPOCHS" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" \
            --text_feature_dir "$TEXT_FEATURE_DIR" \
            --eeg_data_dir "$EEG_DATA_DIR" \
            --device "$DEVICE" \
            --output_dir "$RUN_DIR" \
            --selected_channels "${SELECTED_CHANNELS[@]}" \
            --img_l2norm \
            --projector "$PROJECTOR" \
            --feature_dim "$FEATURE_DIM" \
            --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
            --data_average \
            --save_weights \
            --multi_positive_loss \
            --grouped_batch_sampler \
            --samples_per_image 9 \
            --seed "$SEED" \
            $EXTRA_ARGS
    done

    python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
    upsert_average_row "$CONFIG_NAME" "$RUN_DIR"
done

echo "Mixup regularization sweep completed: ${SESSION_SUMMARY}"
