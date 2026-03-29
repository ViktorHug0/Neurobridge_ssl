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
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-factorized_adv_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/factorized_adv_summary.csv"
SEED="${SEED:-2025}"

mkdir -p "$RUN_ROOT"

SUBSET_CHANNELS=() # ("P7" "P5" "P3" "P1" "Pz" "P2" "P4" "P6" "P8" "PO7" "PO3" "POz" "PO4" "PO8" "O1" "Oz" "O2")

CONFIG_NAMES=(
    # "frozen_pre12_ssl005_a01"
    # "frozen_pre12_ssl01_a01"
    # "frozen_pre16_ssl01_a01_h512"
    # "frozen_pre16_ssl02_a03"
    # "frozen_pre20_ssl01_a03"
    "factorized_adv01_ortho001"
    "factorized_adv02_ortho001"
    "factorized_adv02_ortho001_recon005"
    "factorized_adv01_ortho0_recon01"
    "factorized_c512_s256_adv02_recon005"
)
# For architecture frozen_adapter,try all combinations of the following :
# pretrain_epochs 15, 25, 35
# ssl_lambda 0.01, 0.05, 0.1, 0.5, 1.0
# adapter_hidden_dim 128, 256, 512
# adapter_alpha 0.1, 0.3, 0.5, 1.0

# For architecture factorized_adv,try all combinations of the following :
# ssl_lambda 0.01, 0.05, 0.1, 0.5, 
# content_dim 128, 256, 512
# style_dim 128, 256, 512
# subject_loss_lambda 0.1, 0.5, 1.0
# adv_subject_loss_lambda 0.1, 0.5, 1.0
# ortho_lambda 0.01, 0.05, 0.1
# recon_lambda 0.01, 0.05, 0.1

CONFIG_ARGS=(
    # "--architecture frozen_adapter --pretrain_epochs 20 --ssl_lambda 1 --adapter_hidden_dim 64 --adapter_alpha 0.1"
    # "--architecture frozen_adapter --pretrain_epochs 12 --ssl_lambda 0.1 --adapter_hidden_dim 256 --adapter_alpha 0.1"
    # "--architecture frozen_adapter --pretrain_epochs 16 --ssl_lambda 0.1 --adapter_hidden_dim 512 --adapter_alpha 0.1"
    # "--architecture frozen_adapter --pretrain_epochs 16 --ssl_lambda 0.2 --adapter_hidden_dim 256 --adapter_alpha 0.3"
    # "--architecture frozen_adapter --pretrain_epochs 20 --ssl_lambda 0.1 --adapter_hidden_dim 256 --adapter_alpha 0.3"
    "--architecture factorized_adv --ssl_lambda 0.001 --content_dim 128 --style_dim 128 --subject_loss_lambda 0.01 --adv_subject_loss_lambda 0.01 --ortho_lambda 0.01 --recon_lambda 0.0"
    "--architecture factorized_adv --ssl_lambda 0.1 --content_dim 256 --style_dim 256 --subject_loss_lambda 1.0 --adv_subject_loss_lambda 0.2 --ortho_lambda 0.01 --recon_lambda 0.0"
    "--architecture factorized_adv --ssl_lambda 0.1 --content_dim 256 --style_dim 256 --subject_loss_lambda 1.0 --adv_subject_loss_lambda 0.2 --ortho_lambda 0.01 --recon_lambda 0.05"
    "--architecture factorized_adv --ssl_lambda 0.1 --content_dim 256 --style_dim 256 --subject_loss_lambda 1.0 --adv_subject_loss_lambda 0.1 --ortho_lambda 0.0 --recon_lambda 0.1"
    "--architecture factorized_adv --ssl_lambda 0.1 --content_dim 512 --style_dim 256 --subject_loss_lambda 1.0 --adv_subject_loss_lambda 0.2 --ortho_lambda 0.01 --recon_lambda 0.05"
)

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

ARCH="TSConv"
CURRENT_CHANNELS=("${SUBSET_CHANNELS[@]}")

for c_idx in "${!CONFIG_NAMES[@]}"
do
    CONFIG_NAME="${CONFIG_NAMES[$c_idx]}"
    EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
    CONFIG_RUN_DIR="${RUN_ROOT}/${CONFIG_NAME}_seed${SEED}"
    mkdir -p "$CONFIG_RUN_DIR"

    echo "=========================================================="
    echo "Running DG Sweep: ${CONFIG_NAME}"
    echo "Args: ${EXTRA_ARGS}"
    echo "=========================================================="

    for SUB_ID in {1..1}
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
            --seed "$SEED" \
            $EXTRA_ARGS

        python3 compute_avg_results.py --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
    done

    append_average_row "$CONFIG_NAME" "$CONFIG_RUN_DIR"
done

echo "Architecture sweep completed: ${UNIFIED_CSV}"
