#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-${REPO_ROOT}/data/things_eeg/image_feature}"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"

# Put the alljoined EEG data here by default, or override with:
EEG_DATA_DIR="${EEG_DATA_DIR:-${REPO_ROOT}/data/alljoined/preprocessed_eeg}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/alljoined/inter-subjects}"

SEEDS="${SEEDS:-3300 3301 3302}"
SUBJECT_IDS="${SUBJECT_IDS:-1 2 3 4 5 6 7 8 9 10}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-${SUBJECT_IDS}}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.5}"
TSCONV_FEATURE_DIM="${TSCONV_FEATURE_DIM:-64}"

# InternViT-6B_layer28_mean_8bit currently has width 3200, which the direct
# projector baseline must match.
EEGPROJECT_FEATURE_DIM="${EEGPROJECT_FEATURE_DIM:-3200}"

RUN_TAG="${RUN_TAG:-alljoined_compare_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/comparison_summary.csv"

mkdir -p "$RUN_ROOT"

SUBSET_CHANNELS=() # Example: "P7" "P5" "P3" "P1" "Pz" "P2" "P4" "P6" "P8" "PO7" "PO3" "POz" "PO4" "PO8" "O1" "Oz" "O2"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a SUBJECT_ID_ARR <<< "$SUBJECT_IDS"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

CONFIG_NAMES=(
    "eegproject_internvit_direct"
    "tsconv_fd${TSCONV_FEATURE_DIM}_mixup_raw_pairwise_linear_a${MIXUP_ALPHA//./p}"
)
CONFIG_ARGS=(
    "--eeg_encoder_type EEGProject --projector direct --feature_dim ${EEGPROJECT_FEATURE_DIM} --subject_mixup_mode none"
    "--eeg_encoder_type TSConv --projector linear --feature_dim ${TSCONV_FEATURE_DIM} --eeg_backbone_dim ${TSCONV_FEATURE_DIM} --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha ${MIXUP_ALPHA}"
)

echo "----------------------------------------------------------"
echo "Alljoined inter-subject comparison"
echo "RUN_ROOT:           ${RUN_ROOT}"
echo "EEG_DATA_DIR:       ${EEG_DATA_DIR}"
echo "IMAGE_FEATURE_DIR:  ${IMAGE_FEATURE_DIR}"
echo "SEEDS:              ${SEED_ARR[*]}"
echo "SUBJECT_IDS:        ${SUBJECT_ID_ARR[*]}"
echo "HELD_OUT_SUBJECTS:  ${HELD_OUT_SUBJECT_ARR[*]}"
echo "CONFIGS:            ${CONFIG_NAMES[*]}"
echo "----------------------------------------------------------"

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
        echo "Running alljoined comparison: ${CONFIG_NAME} (Seed ${SEED})"
        echo "Args: ${EXTRA_ARGS}"
        echo "=========================================================="

        for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
        do
            OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
            TRAIN_IDS=""
            for TRAIN_SUB_ID in "${SUBJECT_ID_ARR[@]}"
            do
                if [ "$TRAIN_SUB_ID" -ne "$SUB_ID" ]; then
                    TRAIN_IDS+="$TRAIN_SUB_ID "
                fi
            done

            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python3 "${REPO_ROOT}/train.py" \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --learning_rate "$LEARNING_RATE" \
                --output_name "$OUTPUT_NAME" \
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
                --data_average \
                --save_weights \
                --multi_positive_loss \
                --grouped_batch_sampler \
                --samples_per_image 9 \
                --seed "$SEED" \
                $EXTRA_ARGS
        done

        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
        append_average_row "${CONFIG_NAME}_seed${SEED}" "$CONFIG_RUN_DIR"
    done
done

echo "Alljoined comparison completed: ${UNIFIED_CSV}"
