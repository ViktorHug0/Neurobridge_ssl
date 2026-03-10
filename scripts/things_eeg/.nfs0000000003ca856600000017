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
LEARNING_RATE=1e-4
NUM_EPOCHS=40
NUM_WORKERS=4
SELECTED_CHANNELS=() # "P7" "P5" "P3" "P1" "Pz" "P2" "P4" "P6" "P8" "PO7" "PO3" "POz" "PO4" "PO8" "O1" "Oz" "O2")
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="./results/things_eeg/inter-subjects"

# Default extra arguments (can be overridden by environment variable EXTRA_ARGS)
# Baseline run should set EXTRA_ARGS to an empty string.
DEFAULT_EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
EXTRA_ARGS=${EXTRA_ARGS-$DEFAULT_EXTRA_ARGS}

# Default seed (can be overridden by environment variable SEED)
SEED=${SEED:-2025}

# Create a dedicated sub-folder for this inter-subject run
RUN_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
RUN_DIR="${OUTPUT_DIR}/${RUN_TIMESTAMP}_seed${SEED}"
mkdir -p "$RUN_DIR"

for SUB_ID in {1..10}
do
    OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."

    VAL_ID=1
    if [ "$SUB_ID" -eq 1 ]; then
        VAL_ID=2
    fi

    TRAIN_IDS=""
    for i in {1..10}
    do
        if [ "$i" -ne "$SUB_ID" ] && [ "$i" -ne "$VAL_ID" ]; then
            TRAIN_IDS+="$i "
        fi
    done

    python train.py \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $TRAIN_IDS \
        --val_subject_ids $VAL_ID \
        --test_subject_ids $SUB_ID \
        --strict_dg \
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
        --feature_dim "$FEATURE_DIM" \
        --data_average \
        --save_weights \
        $EXTRA_ARGS \
        --seed "$SEED";

    # Dynamically update the summary CSV after each subject run
    python compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
done
