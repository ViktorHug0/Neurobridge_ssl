#!/bin/bash
# Run 1: this repo's TSConv recipe, this repo's EEG preprocessing, the challenge's DINOv2 targets.
#
# Pooled all-10 subjects (what the challenge harness does), 63 channels, our training recipe.
# --projector direct --feature_dim 1536 is not a style choice: the Codabench grader ranks the
# model's raw 1536-D output against raw DINOv2 candidates and cannot apply a learned image
# projector, so any other projector makes the checkpoint unscoreable under the competition rules.
set -e
trap 'echo "Script Error"' ERR

cd "$(dirname "$0")/.."

IMAGE_FEATURE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"
OUTPUT_DIR="./results/things_eeg/neurips_track1/run1_tsconv_pooled63"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-33}"

# Checkpoint selection never touches the test split: the leaderboard number has to be honest,
# and our usual "best top1 acc" convention selects on test. Concept holdout keeps all 10
# subjects in training, unlike --val_subject_id.
.venv/bin/python train.py \
    --train_subject_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_subject_ids  1 2 3 4 5 6 7 8 9 10 \
    --output_name "pooled63" \
    --output_dir "$OUTPUT_DIR" \
    --eeg_data_dir "$EEG_DATA_DIR" \
    --image_feature_dir "$IMAGE_FEATURE_DIR" \
    --text_feature_dir "" \
    --eeg_encoder_type TSConv \
    --projector direct \
    --feature_dim 1536 \
    --time_window 0 250 \
    --data_average \
    --multi_positive_loss \
    --grouped_batch_sampler \
    --samples_per_image 10 \
    --img_l2norm \
    --softplus \
    --batch_size 1024 \
    --learning_rate 3e-4 \
    --num_epochs 50 \
    --val_concept_ratio 0.1 \
    --select_best_on val \
    --save_weights \
    --seed "$SEED" \
    --device "$DEVICE"

RUN_DIR=$(ls -d "$OUTPUT_DIR"/*-pooled63 | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device "$DEVICE"
