#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"

DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=3e-4
NUM_EPOCHS=50
NUM_WORKERS=4
PROJECTOR="linear"

OUTPUT_DIR_BASE=${OUTPUT_DIR:-"./results/things_eeg/inter-subjects"}

# Configuration sweep: feature_dim values (projector size)
FEATURE_DIM_VALUES=(16 32 64 128 256 512 1024)
SEEDS=(3300 3301 3302)

# Create a dedicated session folder for this entire execution
SESSION_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
SESSION_DIR="${OUTPUT_DIR_BASE}/projector_sweep_${SESSION_TIMESTAMP}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "$SESSION_DIR"

# Baseline extra args as used in the mixup baseline runs
BASE_EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 9"

for SEED in "${SEEDS[@]}"
do
    for DIM in "${FEATURE_DIM_VALUES[@]}"
    do
        CONFIG_NAME="featdim_${DIM}_seed${SEED}"
        RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
        mkdir -p "$RUN_DIR"

        echo "##########################################################"
        echo "Running Config: $CONFIG_NAME"
        echo "Seed: $SEED, Dim: $DIM"
        echo "##########################################################"

        for SUB_ID in {1..10}
        do
            OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
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
                --test_subject_ids $SUB_ID \
                --select_best_on test \
                --softplus \
                --num_epochs "$NUM_EPOCHS" \
                --image_feature_dir "$IMAGE_FEATURE_DIR" \
                --text_feature_dir "$TEXT_FEATURE_DIR" \
                --eeg_data_dir "$EEG_DATA_DIR" \
                --device "$DEVICE"  \
                --output_dir "$RUN_DIR" \
                --img_l2norm \
                --projector "$PROJECTOR" \
                --feature_dim "$DIM" \
                --eeg_backbone_dim "$DIM" \
                --data_average \
                --save_weights \
                --seed "$SEED" \
                $BASE_EXTRA_ARGS;

            # Dynamically update the summary CSV after each subject run
            python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
        done

        # Append the Average row to the session summary
        echo "Collecting average results for $CONFIG_NAME into $SESSION_SUMMARY..."
        python3 - "$SESSION_SUMMARY" "$CONFIG_NAME" "$RUN_DIR" <<'PY'
import pandas as pd
import os
import sys

summary_file, config_name, run_dir = sys.argv[1:4]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")

if os.path.exists(avg_csv):
    df = pd.read_csv(avg_csv)
    avg_row = df[df['sub'] == 'Average'].copy()
    if not avg_row.empty:
        avg_row.insert(0, 'config', config_name)
        
        write_header = not os.path.exists(summary_file)
        avg_row.to_csv(summary_file, mode='a', header=write_header, index=False)
PY
    done
done

echo "Projector size sweep completed: $SESSION_SUMMARY"
