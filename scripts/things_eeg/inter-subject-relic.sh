#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=3e-4
NUM_EPOCHS=60
NUM_WORKERS=4
SELECTED_CHANNELS=() # 'P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
PROJECTOR="linear"
# RELIC sweep below; contrastive temperature fixed via INIT_TEMPERATURE (default 0.07, train.py default).
INIT_TEMPERATURE="${INIT_TEMPERATURE:-0.07}"
FEATURE_DIM=64
EEG_BACKBONE_DIM=64
OUTPUT_DIR_BASE=${OUTPUT_DIR:-"./results/things_eeg/inter-subjects"}

# Configuration sweep: relic_lambda (prediction-space same-image cross-subject consistency weight).
# Override with RELIC_LAMBDA_VALUES env.
RELIC_LAMBDA_VALUES=(${RELIC_LAMBDA_VALUES:-0.03 0.05 0.07 0.1 0.12 0.15 0.18 0.20})
CONFIG_NAMES=()
CONFIG_ARGS=()

for val in "${RELIC_LAMBDA_VALUES[@]}"
do
    CONFIG_NAMES+=("relic_lambda_${val}")
    CONFIG_ARGS+=("--init_temperature ${INIT_TEMPERATURE} --feature_dim ${FEATURE_DIM} \
    --eeg_backbone_dim ${EEG_BACKBONE_DIM} --relic_lambda ${val} --ssl_lambda 0.0 --multi_positive_loss \
    --grouped_batch_sampler --samples_per_image 2")
done

# Default seed (can be overridden by environment variable SEED)
SEED=${SEED:-9099}

# Create a dedicated session folder for this entire execution
SESSION_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
SESSION_DIR="${OUTPUT_DIR_BASE}/20260401-151005_session_seed9099" # ${SESSION_TIMESTAMP}_session_seed${SEED}
SESSION_SUMMARY="${SESSION_DIR}/session_summary.csv"
mkdir -p "$SESSION_DIR"

for c_idx in "${!CONFIG_NAMES[@]}"
do
    CONFIG_NAME="${CONFIG_NAMES[$c_idx]}"
    EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"

    echo "##########################################################"
    echo "Running Config: $CONFIG_NAME"
    echo "Args: $EXTRA_ARGS"
    echo "##########################################################"

    # Create a sub-folder for this specific configuration inside the session folder
    RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
    mkdir -p "$RUN_DIR"

    for SUB_ID in 1 2 3 4 5 6 7 8 9 10
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

        python3 train.py \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --learning_rate "$LEARNING_RATE" \
            --output_name "$OUTPUT_NAME" \
            --eeg_encoder_type "$EEG_ENCODER_TYPE" \
            --train_subject_ids $TRAIN_IDS \
            --test_subject_ids $SUB_ID \
            --softplus \
            --num_epochs "$NUM_EPOCHS" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" \
            --text_feature_dir "$TEXT_FEATURE_DIR" \
            --eeg_data_dir "$EEG_DATA_DIR" \
            --device "$DEVICE"  \
            --output_dir "$RUN_DIR" \
            --selected_channels "${SELECTED_CHANNELS[@]}" \
            --img_l2norm \
            --projector "$PROJECTOR" \
            --data_average \
            --save_weights \
            $EXTRA_ARGS \
            --seed "$SEED";

        # Dynamically update the summary CSV after each subject run
        python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
    done

    # After finishing all subjects for this config, append the Average row to the session summary
    echo "Collecting average results for $CONFIG_NAME into $SESSION_SUMMARY..."
    python3 -c "
import pandas as pd
import os
import sys

run_dir = sys.argv[1]
config_name = sys.argv[2]
session_summary_path = sys.argv[3]

inter_summary_path = os.path.join(run_dir, 'inter_subject_summary.csv')
if os.path.exists(inter_summary_path):
    try:
        df = pd.read_csv(inter_summary_path)
        avg_row = df[df['sub'] == 'Average'].copy()
        if not avg_row.empty:
            avg_row.insert(0, 'config', config_name)

            header = not os.path.exists(session_summary_path)
            avg_row.to_csv(session_summary_path, mode='a', index=False, header=header)
            print(f'Successfully added {config_name} to {session_summary_path}')
        else:
            print(f'Warning: No Average row found in {inter_summary_path}')
    except Exception as e:
        print(f'Error processing {inter_summary_path}: {e}')
else:
    print(f'Warning: {inter_summary_path} not found')
" "$RUN_DIR" "$CONFIG_NAME" "$SESSION_SUMMARY"
done
