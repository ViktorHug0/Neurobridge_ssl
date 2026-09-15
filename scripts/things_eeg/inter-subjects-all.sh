#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="./data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="RN50"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
# Use NICE-EEG preprocessed files directly (no copying/renaming required).
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="EEGProject"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=50
# SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2') # "Oz" "O1" "O2" "POz" "PO3" "PO4" "PO7" "PO8" "Pz" "P1" "P2" "P3" "P4" "P5" "P6" "P7" "P8" "TP7" "TP8" "T7" "T8" "FT7" "FT8")
# EEG_ENCODER_TYPE="TSConv"
# BATCH_SIZE=1024
# LEARNING_RATE=1e-4
# NUM_EPOCHS=50
# SELECTED_CHANNELS=() ('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="./results/things_eeg/inter-subjects-all"
NUM_WORKERS="4" # "$(nproc)"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${OUTPUT_DIR}/last_run.txt"

TRAIN_IDS="1 2 3 4 5 6 7 8 9 10"
TEST_IDS="1 2 3 4 5 6 7 8 9 10"

python train.py \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --output_name "all-subjects" \
    --eeg_encoder_type "$EEG_ENCODER_TYPE" \
    --train_subject_ids $TRAIN_IDS \
    --test_subject_ids $TEST_IDS \
    --softplus \
    --num_epochs "$NUM_EPOCHS" \
    --image_feature_dir "$IMAGE_FEATURE_DIR" \
    --text_feature_dir "$TEXT_FEATURE_DIR" \
    --eeg_data_dir "$EEG_DATA_DIR" \
    --device "$DEVICE" \
    --output_dir "$RUN_DIR" \
    --selected_channels "${SELECTED_CHANNELS[@]}" \
    --num_workers "$NUM_WORKERS" \
    --img_l2norm \
    --projector "$PROJECTOR" \
    --feature_dim "$FEATURE_DIM" \
    --data_average \
    --save_weights \
    --multi_positive_loss \
    --seed 19992765 \
    --image_aug \
    --aug_image_feature_dirs "./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic" \
    --eeg_aug \
    --eeg_aug_type "smooth" \
    --image_test_aug