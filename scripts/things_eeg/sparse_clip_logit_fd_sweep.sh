#!/bin/bash
# Sparse CLIP-style sweep: ReLU feature_dim x logit_scale_max x eeg_l2norm on|off.
# Recipe: bb64, multipos, img_l2norm, learnable temperature (--t_learnable).
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature}"
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
SEED="${SEED:-3300}"
EEG_BACKBONE_DIM_FIXED="${EEG_BACKBONE_DIM_FIXED:-64}"

FEATURE_DIM_VALUES="${FEATURE_DIM_VALUES:-128 512 2048 8192}"
# Prior ReLU runs learned logit scales around 3.8-4.4 without EEG norm and 4.9-5.0 with EEG norm.
# "none" keeps the uncapped baseline; numeric entries are passed as --logit_scale_max.
LOGIT_SCALE_MAX_VALUES="${LOGIT_SCALE_MAX_VALUES:-none  1.0 2.5 3.5 4.5}"
# 1 = pass --eeg_l2norm, 0 = omit flag
EEG_L2NORM_VALUES="${EEG_L2NORM_VALUES:-0 1}"
SUBJECT_IDS="${SUBJECT_IDS:-1}"

read -r -a FEATURE_DIM_ARR <<< "$FEATURE_DIM_VALUES"
read -r -a LOGIT_SCALE_MAX_ARR <<< "$LOGIT_SCALE_MAX_VALUES"
read -r -a EEG_L2NORM_ARR <<< "$EEG_L2NORM_VALUES"
read -r -a SUBJECT_ARR <<< "$SUBJECT_IDS"

OUTPUT_DIR_BASE="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subject-sparse}"
SESSION_TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
SESSION_DIR="${SESSION_DIR:-${OUTPUT_DIR_BASE}/sparse_clip_relu_fd_logit_bb${EEG_BACKBONE_DIM_FIXED}_seed${SEED}_${SESSION_TIMESTAMP}}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "$SESSION_DIR"

BASE_EXTRA_ARGS="--eeg_backbone_dim ${EEG_BACKBONE_DIM_FIXED} --t_learnable --projector_activation relu"

echo "=============================================================="
echo "Sparse CLIP ReLU fd x logit_scale_max x eeg_l2norm sweep | bb=${EEG_BACKBONE_DIM_FIXED}"
echo "SESSION_DIR: $SESSION_DIR"
echo "FEATURE_DIM_VALUES: ${FEATURE_DIM_ARR[*]}"
echo "LOGIT_SCALE_MAX_VALUES: ${LOGIT_SCALE_MAX_ARR[*]}"
echo "EEG_L2NORM_VALUES: ${EEG_L2NORM_ARR[*]} (1=on, 0=off)"
echo "SUBJECT_IDS: ${SUBJECT_ARR[*]}"
echo "SEED: $SEED | NUM_EPOCHS: $NUM_EPOCHS | t_learnable: yes | projector_activation: relu"
echo "=============================================================="

for DIM in "${FEATURE_DIM_ARR[@]}"
do
    for LOGIT_SCALE_MAX in "${LOGIT_SCALE_MAX_ARR[@]}"
    do
        if [ "$LOGIT_SCALE_MAX" = "none" ]; then
            LOGIT_TAG="logit_uncapped"
            LOGIT_ARGS=""
        else
            LOGIT_TAG="logitmax_${LOGIT_SCALE_MAX//./p}"
            LOGIT_ARGS="--logit_scale_max ${LOGIT_SCALE_MAX}"
        fi

        for EEG_L2 in "${EEG_L2NORM_ARR[@]}"
        do
            if [ "$EEG_L2" = "1" ]; then
                EEG_L2_TAG="eegnorm"
                EEG_L2_ARGS="--eeg_l2norm"
            elif [ "$EEG_L2" = "0" ]; then
                EEG_L2_TAG="noeegnorm"
                EEG_L2_ARGS=""
            else
                echo "EEG_L2NORM_VALUES entries must be 0 or 1, got: $EEG_L2" >&2
                exit 1
            fi

            CONFIG_NAME="sparse_fd${DIM}_relu_${LOGIT_TAG}_${EEG_L2_TAG}_seed${SEED}"
            RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
            mkdir -p "$RUN_DIR"

            echo "##########################################################"
            echo "Config: $CONFIG_NAME | fd=$DIM relu logit_scale_max=$LOGIT_SCALE_MAX eeg_l2norm=$EEG_L2"
            echo "##########################################################"

            for SUB_ID in "${SUBJECT_ARR[@]}"
            do
                OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
                if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
                    echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
                    continue
                fi
                echo "Training subject ${SUB_ID} for $CONFIG_NAME..."

                TRAIN_IDS=""
                for i in {1..10}
                do
                    if [ "$i" -ne "$SUB_ID" ]; then
                        TRAIN_IDS+="$i "
                    fi
                done

                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                python3 train.py \
                    --batch_size "$BATCH_SIZE" \
                    --num_workers "$NUM_WORKERS" \
                    --learning_rate "$LEARNING_RATE" \
                    --output_name "$OUTPUT_NAME" \
                    --eeg_encoder_type "$EEG_ENCODER_TYPE" \
                    --train_subject_ids $TRAIN_IDS \
                    --test_subject_ids "$SUB_ID" \
                    --select_best_on test \
                    --softplus \
                    --num_epochs "$NUM_EPOCHS" \
                    --image_feature_dir "$IMAGE_FEATURE_DIR" \
                    --text_feature_dir "$TEXT_FEATURE_DIR" \
                    --eeg_data_dir "$EEG_DATA_DIR" \
                    --device "$DEVICE" \
                    --output_dir "$RUN_DIR" \
                    --img_l2norm \
                    $EEG_L2_ARGS \
                    --projector "$PROJECTOR" \
                    --feature_dim "$DIM" \
                    --data_average \
                    --save_weights \
                    --seed "$SEED" \
                    $BASE_EXTRA_ARGS \
                    $LOGIT_ARGS
            done

            python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"

            echo "Collecting average results for $CONFIG_NAME into $SESSION_SUMMARY..."
            python3 - "$SESSION_SUMMARY" "$CONFIG_NAME" "$RUN_DIR" <<'PY'
import os
import sys

import pandas as pd

summary_file, config_name, run_dir = sys.argv[1:4]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")

if os.path.exists(avg_csv):
    df = pd.read_csv(avg_csv)
    avg_row = df[df["sub"] == "Average"].copy()
    if not avg_row.empty:
        avg_row.insert(0, "config", config_name)
        write_header = not os.path.exists(summary_file)
        avg_row.to_csv(summary_file, mode="a", header=write_header, index=False)
PY
        done
    done
done

echo "Sparse CLIP ReLU fd x logit_scale_max x eeg_l2norm sweep completed: $SESSION_SUMMARY"
