#!/bin/bash
# SAMGA reference code, their inter.sh recipe MINUS --eeg_aug smooth / --frozen_eeg_prior.
# Those two make their loader eagerly smooth the whole training set in a Python loop (16.5k calls
# per subject, no disk cache), which dominates wall-clock; and this repo's smoothing sweep never
# beat no-aug at any strength, so dropping them is cheap here.
# Usage: ./run_samga_nosmooth.sh <arm_name> <device> <subject...>
set -u
REF=/nasbrain/p20fores/SAMGA_ref
VENV=/nasbrain/p20fores/Neurobridge_SSL/.venv
IMG=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/internvit_multilayer_src
EEG=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/
ARM=$1; DEV=$2; shift 2
OUT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25_samga/$ARM
mkdir -p "$OUT"; cd "$REF" || exit 1

for S in "$@"; do
  NAME=$(printf "sub-%02d" "$S")
  if compgen -G "${OUT}/*-${NAME}/result.csv" > /dev/null; then echo "[$NAME] done, skip"; continue; fi
  TRAIN_IDS=$(seq 1 10 | grep -vw "$S" | tr '\n' ' ')
  echo "=== $ARM $NAME on $DEV ==="
  "$VENV/bin/python" train.py \
    --batch_size 1024 --learning_rate 1e-4 --num_epochs 30 \
    --output_name "$NAME" --output_dir "$OUT" \
    --eeg_encoder_type TSConv \
    --train_subject_ids $TRAIN_IDS --test_subject_ids "$S" \
    --softplus --img_l2norm --data_average --save_weights \
    --image_feature_dir "$IMG" --text_feature_dir '' --eeg_data_dir "$EEG" \
    --eeg_feature_dim 1024 --projector linear --feature_dim 512 \
    --stage1_mmd_start 0.9 --stage1_mmd_end 0.5 \
    --use_multilayer_router --layer_ids 23 25 28 31 33 \
    --layer_prior_center 28 --layer_prior_strength 1.0 --router_eval_mode global \
    --device "$DEV" --seed 3300
done
echo "=== $ARM DONE $(date +%H:%M:%S) ==="
