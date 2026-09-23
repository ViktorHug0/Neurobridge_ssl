#!/bin/bash
# Reproduce the 0.2911 REVE submission as closely as this repo allows.
#
# Reference: Neurips_challenge job 54022, resolved config at
#   results/neuralbench.main.Experiment.run,1/...818-566da833/config.yaml
# plus submissions/reve_full/submission.py, which holds the 200 Hz inference path.
# NOTE: track1_reve.yaml is NOT that recipe -- it says batch_size 64 and omits gradient_clip_val.
#
# Two data paths, deliberately asymmetric, as the reference run was:
#   train/val   native 200 Hz, 0.5-99.5 Hz, no notch, no baseline, StandardScaler, clamp 15
#   scoring     official 120 Hz tensors, linearly interpolated to 200 inside the wrapper
# score_track1.py already does the second: it serves OFFICIAL_EEG_DIR and resamples to the
# model's native n_times, which is exactly submission.py's F.interpolate(x, size=200).
#
# Known deviations, neither reproducible here:
#   - validation split: theirs holds out 20% of whole recordings (valid_split_by: timeline);
#     session identity is not preserved in our cached arrays, so we hold out concepts instead.
#   - test query list: the reference scorer counted 162,240 epochs, ours counts 160,000.
set -e
trap 'echo "REVE REPRO FAILED"' ERR

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/nasbrain/p20fores/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

OUT_DIR="${OUT_DIR:-./results/things_eeg/neurips_track1/reve_repro}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-33}"
BATCH="${BATCH:-128}"
PATIENCE="${PATIENCE:-5}"

.venv/bin/python train.py \
    --train_subject_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_subject_ids  1 2 3 4 5 6 7 8 9 10 \
    --output_name "REVE" \
    --output_dir "$OUT_DIR" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_200Hz_reve \
    --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1 \
    --text_feature_dir "" \
    --eeg_encoder_type REVE \
    --reve_input_normalization none \
    --projector direct \
    --feature_dim 1536 \
    --time_window 0 200 \
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
    --eval_unique_gallery \
    --codabench_eval_every 5 \
    --early_stop_patience "$PATIENCE" \
    --save_weights \
    --seed "$SEED" \
    --device "$DEVICE" \
    "$@"

RUN_DIR=$(ls -d "${OUT_DIR}"/*-REVE | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device "$DEVICE"
echo "scored: $RUN_DIR/track1_score.json"
