#!/bin/bash
# REVE full fine-tune, NeuralBench Track-1 recipe, trained on the input the grader actually serves.
#
# Two differences from the challenge repo's 0.29 run, both deliberate:
#   1. It explores past the reference run's early-stopping point. That run (job 54022) peaked at
#      *epoch 6* on a within-batch 128-way val top-5 and was stopped by patience 5; the 18 h
#      Slurm TIMEOUT hit the surrounding job, not the training. Patience is raised here so we
#      see the whole curve, and --select_best_on val still keeps the best epoch, so this can
#      only help. Do NOT describe 0.29 as a truncated checkpoint -- it converged by its own rule.
#   2. No train/test preprocessing handicap. Theirs trained on 200 Hz / 0.5-99.5 Hz /
#      StandardScaler and was then served 120 Hz / 0.1-75 Hz + notch / RobustScaler tensors,
#      interpolated. We train on the grader's own 120 Hz data, upsampled to REVE's 200 samples
#      inside the wrapper -- the identical transform at train and inference.
#
# The wrapper also restandardises per trial and clips at 15 SD, matching REVE's pretraining
# convention (braindecode reve.py), applied on both sides so it adds no new mismatch.
#
# Everything else is their recipe: ClipLoss(symmetric=False), fixed temperature, AdamW 1e-4 /
# wd 0.05, OneCycleLR per optimizer step, grad clip 1.0, 40 epochs.
set -e
trap 'echo "REVE RUN FAILED"' ERR

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/nasbrain/p20fores/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

OUT_DIR="./results/things_eeg/neurips_track1/reve${TAG:+_$TAG}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-33}"
BATCH="${BATCH:-256}"

.venv/bin/python train.py \
    --train_subject_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_subject_ids  1 2 3 4 5 6 7 8 9 10 \
    --output_name "REVE" \
    --output_dir "$OUT_DIR" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1 \
    --text_feature_dir "" \
    --eeg_encoder_type REVE \
    --projector direct \
    --feature_dim 1536 \
    --time_window 0 120 \
    --data_average \
    --img_l2norm \
    --init_temperature 1.0 \
    --asymmetric_loss \
    --grad_clip 1.0 \
    --lr_scheduler onecycle \
    --onecycle_pct_start 0.1 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --batch_size "$BATCH" \
    --num_epochs 40 \
    --val_concept_ratio 0.1 \
    --select_best_on val \
    --select_best_metric top5 \
    --save_weights \
    --seed "$SEED" \
    --device "$DEVICE" \
    "$@"

RUN_DIR=$(ls -d "${OUT_DIR}"/*-REVE | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device "$DEVICE"
echo "scored: $RUN_DIR/track1_score.json"
