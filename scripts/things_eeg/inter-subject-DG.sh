#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-arch_sweep_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/arch_sweep_summary.csv"
SEED="${SEED:-2025}"

mkdir -p "$RUN_ROOT"

ARCHS=("EEGNet" "EEGProject" "TSConv" "EEGTransformer")
SUBSET_CHANNELS=("P7" "P5" "P3" "P1" "Pz" "P2" "P4" "P6" "P8" "PO7" "PO3" "POz" "PO4" "PO8" "O1" "Oz" "O2")

append_average_row() {
    local arch="$1"
    local ch_set="$2"
    local run_dir="$3"
    python3 - "$UNIFIED_CSV" "$arch" "$ch_set" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, arch, ch_set, run_dir = sys.argv[1:5]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "arch": arch,
    "channels": ch_set,
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

for ARCH in "${ARCHS[@]}"
do
    for CH_SET_TYPE in "subset" "all"
    do
        if [ "$CH_SET_TYPE" == "subset" ]; then
            CURRENT_CHANNELS=("${SUBSET_CHANNELS[@]}")
            CH_LABEL="subset"
        else
            CURRENT_CHANNELS=()
            CH_LABEL="all"
        fi

        FULL_CONFIG_NAME="${ARCH}_${CH_LABEL}"
        CONFIG_RUN_DIR="${RUN_ROOT}/${FULL_CONFIG_NAME}_seed${SEED}"
        mkdir -p "$CONFIG_RUN_DIR"

        echo "=========================================================="
        echo "Running Sweep Config: ${FULL_CONFIG_NAME}"
        echo "Arch: ${ARCH}, Channels: ${CH_LABEL}"
        echo "=========================================================="

        # Adjust batch size for memory-intensive architectures
        CURRENT_BATCH_SIZE="$BATCH_SIZE"
        if [ "$ARCH" == "EEGTransformer" ]; then
            CURRENT_BATCH_SIZE=$((BATCH_SIZE / 4))
            echo "Reducing batch size to $CURRENT_BATCH_SIZE for EEGTransformer"
        fi

        for SUB_ID in {1..5}
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
                --batch_size "$CURRENT_BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --learning_rate "$LEARNING_RATE" \
                --output_name "$OUTPUT_NAME" \
                --eeg_encoder_type "$ARCH" \
                --train_subject_ids $TRAIN_IDS \
                --test_subject_ids "$SUB_ID" \
                --softplus \
                --num_epochs "$NUM_EPOCHS" \
                --image_feature_dir "$IMAGE_FEATURE_DIR" \
                --text_feature_dir "$TEXT_FEATURE_DIR" \
                --eeg_data_dir "$EEG_DATA_DIR" \
                --device "$DEVICE" \
                --output_dir "$CONFIG_RUN_DIR" \
                --selected_channels "${CURRENT_CHANNELS[@]}" \
                --img_l2norm \
                --projector "$PROJECTOR" \
                --feature_dim "$FEATURE_DIM" \
                --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
                --data_average \
                --save_weights \
                --multi_positive_loss \
                --grouped_batch_sampler \
                --samples_per_image 9 \
                --seed "$SEED"

            python3 compute_avg_results.py --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
        done

        append_average_row "$ARCH" "$CH_LABEL" "$CONFIG_RUN_DIR"
    done
done

echo "Architecture sweep completed: ${UNIFIED_CSV}"
