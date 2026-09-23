#!/bin/bash
# R1 and friends: train one architecture on the 120 Hz grader input with the Track-1 recipe,
# then score it with the Codabench protocol.
#
#   run_recipe.sh <arch> <tag> [extra train.py args...]
#
# Recipe differences from sweep_cell.sh, which is the whole point of the experiment:
#   - no --grouped_batch_sampler / --multi_positive_loss (plain random batches, duplicates
#     count as negatives, as NeuralBench's loader does)
#   - ClipLoss(symmetric=False)          -> --asymmetric_loss
#   - fixed logit scale 1.0              -> --init_temperature 1.0, no --softplus
#     (the sweep used softplus(log(1/0.07)) = 2.727, fixed, not learnable)
#   - OneCycleLR per optimizer step, max_lr 1e-3 (was 1e-4; switched 2026-09-23), pct_start 0.1, wd 0.05, grad clip 1.0
#   - 40 epochs instead of 50
#
# Deliberately NOT matched, both flagged in the README:
#   - batch 1024, not 128 (user's call: speed; lr is left at their 1e-4 so this is a confound)
#   - --data_average; NeuralBench trains on single trials. Measured at -0.8 pp for TSConv.
set -e
trap 'echo "RECIPE RUN FAILED: $ARCH/$TAG"' ERR

cd "$(dirname "$0")/.."

ARCH="${1:?usage: run_recipe.sh <arch> <tag> [extra args...]}"
TAG="${2:?usage: run_recipe.sh <arch> <tag> [extra args...]}"
shift 2

OUT_DIR="./results/things_eeg/neurips_track1/recipe/${TAG}/${ARCH}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-33}"
BATCH="${BATCH:-1024}"

.venv/bin/python train.py \
    --train_subject_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_subject_ids  1 2 3 4 5 6 7 8 9 10 \
    --output_name "$ARCH" \
    --output_dir "$OUT_DIR" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1 \
    --text_feature_dir "" \
    --eeg_encoder_type "$ARCH" \
    --projector direct \
    --feature_dim 1536 \
    --time_window 0 120 \
    ${AVERAGE---data_average} \
    --img_l2norm \
    --init_temperature 1.0 \
    --asymmetric_loss \
    --grad_clip 1.0 \
    --lr_scheduler onecycle \
    --onecycle_pct_start 0.1 \
    --learning_rate 1e-3 \
    --weight_decay 0.05 \
    --batch_size "$BATCH" \
    --num_epochs 40 \
    --val_concept_ratio 0.1 \
    --select_best_on val \
    --select_best_metric top5 \
    --eval_unique_gallery \
    --codabench_eval_every 5 \
    --save_weights \
    --seed "$SEED" \
    --device "$DEVICE" \
    "$@"

RUN_DIR=$(ls -d "${OUT_DIR}"/*-"${ARCH}" | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device "$DEVICE"
echo "scored: $RUN_DIR/track1_score.json"
