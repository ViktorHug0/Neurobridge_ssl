#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv30}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-sattc_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_summary.csv"

# Base model recipe to compare evaluation-only SATTC variants against.
BASE_CONFIG_NAME="${BASE_CONFIG_NAME:-baseline}"
BASE_EXTRA_ARGS="${BASE_EXTRA_ARGS:---architecture baseline --ssl_lambda 0.0 --relic_lambda 0.0 --subject_mixup_mode none}"

EVAL_MODES="${EVAL_MODES:-plain_cosine saw saw_csls saw_adacsls saw_adacsls_poe}"
PROJECTORS="${PROJECTORS:-linear}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2}"
SEEDS="${SEEDS:-3300}"

mkdir -p "$RUN_ROOT"

SUBSET_CHANNELS=("P7" "P5" "P3" "P1" "Pz" "P2" "P4" "P6" "P8" "PO7" "PO3" "POz" "PO4" "PO8" "O1" "Oz" "O2")

read -r -a EVAL_MODE_ARR <<< "$EVAL_MODES"
read -r -a PROJECTOR_ARR <<< "$PROJECTORS"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"
read -r -a SEED_ARR <<< "$SEEDS"

CONFIG_NAMES=()
CONFIG_ARGS=()
ARCH="$EEG_ENCODER_TYPE"

for PROJECTOR in "${PROJECTOR_ARR[@]}"; do
    for MODE in "${EVAL_MODE_ARR[@]}"; do
        CONFIG_NAMES+=("${BASE_CONFIG_NAME}_${PROJECTOR}_${MODE}")
        CONFIG_ARGS+=("${BASE_EXTRA_ARGS} --projector ${PROJECTOR} --eval_mode ${MODE}")
    done
done

append_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$UNIFIED_CSV" "$config_name" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, config_name, run_dir = sys.argv[1:4]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "config": config_name,
    "architecture": "",
    "eval_mode": "",
    "top1 acc": "",
    "top5 acc": "",
    "best top1 acc": "",
    "best top5 acc": "",
    "best test loss": "",
    "best epoch": "",
}

if os.path.isfile(summary_path):
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("sub", "").strip().lower() == "average":
                for key in row:
                    if key in r:
                        row[key] = r[key]
                break

write_header = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

CURRENT_CHANNELS=("${SUBSET_CHANNELS[@]}")

for SEED in "${SEED_ARR[@]}"
do
    for c_idx in "${!CONFIG_NAMES[@]}"
    do
        CONFIG_NAME="${CONFIG_NAMES[$c_idx]}"
        EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
        CONFIG_RUN_DIR="${RUN_ROOT}/${CONFIG_NAME}_seed${SEED}"
        mkdir -p "$CONFIG_RUN_DIR"

        echo "=========================================================="
        echo "Running Inter-subject SATTC Sweep: ${CONFIG_NAME} (Seed ${SEED})"
        echo "Args: ${EXTRA_ARGS}"
        echo "=========================================================="

        for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
        do
            OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
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
                --eeg_encoder_type "$ARCH" \
                --train_subject_ids $TRAIN_IDS \
                --test_subject_ids "$SUB_ID" \
                --select_best_on test \
                --softplus \
                --num_epochs "$NUM_EPOCHS" \
                --image_feature_dir "$IMAGE_FEATURE_DIR" \
                --text_feature_dir "$TEXT_FEATURE_DIR" \
                --eeg_data_dir "$EEG_DATA_DIR" \
                --device "$DEVICE" \
                --output_dir "$CONFIG_RUN_DIR" \
                --selected_channels "${CURRENT_CHANNELS[@]}" \
                --img_l2norm \
                --feature_dim "$FEATURE_DIM" \
                --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
                --data_average \
                --save_weights \
                --multi_positive_loss \
                --grouped_batch_sampler \
                --samples_per_image 3 \
                --seed "$SEED" \
                $EXTRA_ARGS
        done

        python3 compute_avg_results.py --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
        append_average_row "${CONFIG_NAME}_seed${SEED}" "$CONFIG_RUN_DIR"
    done
done

echo "SATTC sweep completed: ${UNIFIED_CSV}"
