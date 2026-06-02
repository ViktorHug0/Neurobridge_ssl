#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="./data/things_eeg/image_feature"
IMAGE_FEATURE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-EEGProject}" # "EEGConformer"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=50
SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
PROJECTOR="linear"
FEATURE_DIM=512
EEG_BACKBONE_DIM=1024
OUTPUT_DIR="./results/things_eeg/intra-subjects/TTA_2"

for SUB_ID in {1..10}
do
    OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."
    python train.py \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $SUB_ID \
        --test_subject_ids $SUB_ID \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE"  \
        --output_dir "$OUTPUT_DIR" \
        --selected_channels "${SELECTED_CHANNELS[@]}" \
        --img_l2norm \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
        --data_average \
        --save_weights \
        --seed 2025;
done

python compute_avg_results.py --result_dir "$OUTPUT_DIR";