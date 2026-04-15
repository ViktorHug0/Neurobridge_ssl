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
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
SEED="${SEED:-2099}"

FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4}"
ALL_SUBJECTS="1 2 3 4 5 6 7 8 9 10"

EVAL_MODE="${EVAL_MODE:-saw}"
EVAL_CSLS_K="${EVAL_CSLS_K:-1}"

TRAIN_SAW_SHRINK_VALUES="${TRAIN_SAW_SHRINK_VALUES:-0.8 0.85 0.90}"
TRAIN_SAW_DIAG_VALUES="${TRAIN_SAW_DIAG_VALUES:-0}"

SESSION_TIMESTAMP="20260415-104225"
SESSION_DIR="${OUTPUT_DIR_BASE}/${SESSION_TIMESTAMP}_train_saw_seed${SEED}"
SESSION_SUMMARY="${SESSION_DIR}/session_summary.csv"

sanitize_tag() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "$value"
}

append_average_row() {
    local run_dir="$1"
    local config_name="$2"
    python3 - "$run_dir" "$config_name" "$SESSION_SUMMARY" <<'PY'
import os
import sys
import pandas as pd

run_dir, config_name, session_summary = sys.argv[1:4]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
if not os.path.isfile(summary_path):
    raise FileNotFoundError(summary_path)

df = pd.read_csv(summary_path)
avg = df[df["sub"] == "Average"].copy()
if avg.empty:
    raise ValueError(f"No Average row found in {summary_path}")
avg.insert(0, "config", config_name)
avg.to_csv(session_summary, mode="a", index=False, header=not os.path.exists(session_summary))
PY
}

run_config() {
    local config_name="$1"
    shift
    local extra_args=("$@")
    local run_dir="${SESSION_DIR}/${config_name}"
    mkdir -p "$run_dir"

    echo "##########################################################"
    echo "Running Config: ${config_name}"
    echo "Args: ${extra_args[*]}"
    echo "##########################################################"

    read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"
    read -r -a ALL_SUBJECT_ARR <<< "$ALL_SUBJECT_S"
    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        echo "Training subject ${SUB_ID} for ${config_name}..."

        TRAIN_IDS=""
        # Correctly use all 10 subjects as potential training data
        for i in 1 2 3 4 5 6 7 8 9 10
        do
            if [ "$i" -ne "$SUB_ID" ]; then
                TRAIN_IDS+="$i "
            fi
        done

        python3 "${REPO_ROOT}/train.py" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --learning_rate "$LEARNING_RATE" \
            --output_name "$OUTPUT_NAME" \
            --eeg_encoder_type "$EEG_ENCODER_TYPE" \
            --train_subject_ids $TRAIN_IDS \
            --test_subject_ids "$SUB_ID" \
            --softplus \
            --num_epochs "$NUM_EPOCHS" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" \
            --text_feature_dir "$TEXT_FEATURE_DIR" \
            --eeg_data_dir "$EEG_DATA_DIR" \
            --device "$DEVICE" \
            --output_dir "$run_dir" \
            --img_l2norm \
            --projector "$PROJECTOR" \
            --data_average \
            --save_weights \
            --feature_dim "$FEATURE_DIM" \
            --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
            --ssl_lambda 0 \
            --multi_positive_loss \
            --grouped_batch_sampler \
            --samples_per_image 6 \
            --eval_mode "$EVAL_MODE" \
            --sattc_saw_shrink "${EVAL_SAW_SHRINK:-0.85}" \
            --sattc_csls_k "$EVAL_CSLS_K" \
            --seed "$SEED" \
            "${extra_args[@]}"

        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$run_dir" --output_name "inter_subject_summary.csv"
    done

    append_average_row "$run_dir" "$config_name"
}

read -r -a TRAIN_SAW_SHRINK_ARR <<< "$TRAIN_SAW_SHRINK_VALUES"
read -r -a TRAIN_SAW_DIAG_ARR <<< "$TRAIN_SAW_DIAG_VALUES"

# Complete baseline: no SAW on neither train nor test
EVAL_MODE="plain_cosine" run_config "featdim_${FEATURE_DIM}_complete_baseline"

# Sweep loop (resuming)
for TRAIN_SAW_SHRINK in "${TRAIN_SAW_SHRINK_ARR[@]}"
do
    for TRAIN_SAW_DIAG in "${TRAIN_SAW_DIAG_ARR[@]}"
    do
        TAG="featdim_${FEATURE_DIM}_trainsaw_shrink$(sanitize_tag "$TRAIN_SAW_SHRINK")_no_renorm"
        EXTRA_ARGS=(
            --train_saw
            --train_saw_shrink "$TRAIN_SAW_SHRINK"
            --train_saw_no_renorm
        )
        if [ "$TRAIN_SAW_DIAG" = "1" ]; then
            TAG="${TAG}_diag"
            EXTRA_ARGS+=(--train_saw_diag)
            # Synchronize diag for evaluation
            EVAL_ARGS=(--sattc_saw_diag)
        else
            EVAL_ARGS=()
        fi
        # Match evaluation shrinkage and diag to training
        EVAL_SAW_SHRINK="$TRAIN_SAW_SHRINK" run_config "$TAG" "${EXTRA_ARGS[@]}" "${EVAL_ARGS[@]}"
    done
done

echo "=========================================================="
echo "Train-SAW sweep completed."
echo "Session dir: ${SESSION_DIR}"
echo "Session summary: ${SESSION_SUMMARY}"
echo "=========================================================="
