#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

TEXT_FEATURE_DIR=""
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-512}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-1024}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"

ARCHITECTURES=(
    "ATM"
    "EEGNet"
    "EEGProject"
    "TSConv"
)

# Feature sets: InternViT + ViTH
IMAGE_FEATURE_SETS=(
    "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
    "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-H-14_layer10_mean"
)

SEEDS=(3300)

SESSION_TIMESTAMP=${1:-"20260429-221100"}
SESSION_DIR="${OUTPUT_DIR_BASE}/eeg_encoder_architecture_sweep_mixup_${SESSION_TIMESTAMP}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "${SESSION_DIR}"

# Mixup enabled
BASE_EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"

append_average_row() {
    local summary_file="$1"
    local config_name="$2"
    local architecture="$3"
    local run_dir="$4"
    python3 - "$summary_file" "$config_name" "$architecture" "$run_dir" <<'PY'
import os
import sys
import pandas as pd

summary_file, config_name, architecture, run_dir = sys.argv[1:5]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")

if os.path.exists(avg_csv):
    try:
        df = pd.read_csv(avg_csv)
        avg_row = df[df["sub"] == "Average"].copy()
        if not avg_row.empty:
            if "config" in avg_row.columns:
                avg_row["config"] = config_name
            else:
                avg_row.insert(0, "config", config_name)
            
            avg_row["architecture"] = architecture
            
            write_header = not os.path.exists(summary_file)
            avg_row.to_csv(summary_file, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"Error processing {avg_csv}: {e}")
PY
}

for IMAGE_DIR in "${IMAGE_FEATURE_SETS[@]}"
do
    FEAT_NAME=$(basename "$IMAGE_DIR")
    echo "=========================================================="
    echo "STARTING SWEEP FOR IMAGE FEATURES: $FEAT_NAME (MIXUP)"
    echo "=========================================================="

    for SEED in "${SEEDS[@]}"
    do
        for ARCH in "${ARCHITECTURES[@]}"
        do
            CONFIG_NAME="${FEAT_NAME}_${ARCH}_seed${SEED}_mixup"
            RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
            mkdir -p "${RUN_DIR}"

            if [ -f "$SESSION_SUMMARY" ] && grep -q "^${CONFIG_NAME}," "$SESSION_SUMMARY"; then
                echo "Config ${CONFIG_NAME} already exists, skipping..."
                continue
            fi

            for SUB_ID in {1..10}
            do
                OUTPUT_NAME=$(printf "sub-%02d" "${SUB_ID}")
                SUMMARY_FILE="${RUN_DIR}/inter_subject_summary.csv"
                
                if [ -f "$SUMMARY_FILE" ] && grep -q "^${OUTPUT_NAME}," "$SUMMARY_FILE"; then
                    echo "Subject ${SUB_ID} for ${CONFIG_NAME} already exists, skipping..."
                    continue
                fi

                TRAIN_IDS=""
                for i in {1..10}
                do
                    if [ "${i}" -ne "${SUB_ID}" ]; then
                        TRAIN_IDS+="${i} "
                    fi
                done

                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                python3 train.py \
                    --batch_size "${BATCH_SIZE}" \
                    --num_workers "${NUM_WORKERS}" \
                    --learning_rate "${LEARNING_RATE}" \
                    --output_name "${OUTPUT_NAME}" \
                    --eeg_encoder_type "${ARCH}" \
                    --train_subject_ids ${TRAIN_IDS} \
                    --test_subject_ids "${SUB_ID}" \
                    --select_best_on test \
                    --softplus \
                    --num_epochs "${NUM_EPOCHS}" \
                    --image_feature_dir "${IMAGE_DIR}" \
                    --text_feature_dir "${TEXT_FEATURE_DIR}" \
                    --eeg_data_dir "${EEG_DATA_DIR}" \
                    --device "${DEVICE}" \
                    --output_dir "${RUN_DIR}" \
                    --img_l2norm \
                    --projector "${PROJECTOR}" \
                    --feature_dim "${FEATURE_DIM}" \
                    --eeg_backbone_dim "${EEG_BACKBONE_DIM}" \
                    --data_average \
                    --save_weights \
                    --seed "${SEED}" \
                    ${BASE_EXTRA_ARGS}

                python3 compute_avg_results.py --result_dir "${RUN_DIR}" --output_name "inter_subject_summary.csv"
            done
            python3 compute_avg_results.py --result_dir "${RUN_DIR}" --output_name "inter_subject_summary.csv"
            append_average_row "${SESSION_SUMMARY}" "${CONFIG_NAME}" "${ARCH}" "${RUN_DIR}"
        done
    done
done

echo "InternViT/ViTH mixup sweep completed: ${SESSION_SUMMARY}"
