#!/bin/bash
# Finish the two incomplete arms of the SubjectMix augmentation ablation (R3 Q2).
#
#   base_no_aug         no augmentation at all; the control method_subjectmix was missing.
#                       Matched to method_subjectmix, so it keeps --data_average.
#   within_subject_mix  Beta-mix two of a subject's OWN reps of the same stimulus. Needs
#                       un-averaged reps, so it has no --data_average; that is inherent to
#                       the arm, not a choice here.
#
# Every other flag is copied from the existing runs' train_config.json, so the new subjects
# are directly comparable to the ones already on disk.
#
# Usage:  run_subjectmix_controls.sh <arm> <subject> [<subject> ...]
#   e.g.  run_subjectmix_controls.sh base_no_aug 1 2 3 4 5
#         run_subjectmix_controls.sh within_subject_mix 5 6 7 8 9 10
set -eu
cd "$(dirname "$0")"

ARM=$1; shift
R=./results/things_eeg/subjectmix_rebuttal/$ARM/seed3300

# Averaged-reps regime: matched to method_subjectmix (--data_average, samples_per_image 9).
# Un-averaged regime ("_unavg"): matched to within_subject_mix, which cannot use
# --data_average because mixing a subject's own reps needs the reps to still exist.
# Comparing across the two regimes is not valid, so each has its own no-aug control.
case "$ARM" in
  base_no_aug)          EXTRA="--data_average" ;;
  noise_aug)            EXTRA="--data_average --eeg_aug --eeg_aug_type noise" ;;
  smooth_aug)           EXTRA="--data_average --eeg_aug --eeg_aug_type smooth" ;;
  pairwise_spi3)        EXTRA="--data_average --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 --samples_per_image 3" ;;
  base_no_aug_unavg)    EXTRA="" ;;
  method_subjectmix_unavg) EXTRA="--subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5" ;;
  within_subject_mix)   EXTRA="--subject_mixup_within --within_mix_alpha 0.5" ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac
# samples_per_image is part of the arm for pairwise_spi3, so it must not be set twice
SPI="--samples_per_image 9"
[ "$ARM" = pairwise_spi3 ] && SPI=""

for S in "$@"; do
  TRAIN=$(seq 1 10 | grep -vx "$S" | tr '\n' ' ')
  NAME=$(printf 'sub-%02d' "$S")
  # a directory alone means nothing: sub-05 of within_subject_mix was killed mid-training
  # and left a dir with checkpoints but no result.csv. Only a result.csv counts as done.
  if compgen -G "$R/*-$NAME/result.csv" > /dev/null; then
    echo "== $ARM $NAME already done, skipping" >&2
    continue
  fi
  echo "== $ARM $NAME  (train on $TRAIN)" >&2
  .venv/bin/python train.py \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
    --text_feature_dir '' \
    --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
    --eeg_backbone_dim 1024 --projector linear --softplus --img_l2norm \
    --multi_positive_loss --grouped_batch_sampler $SPI \
    --learning_rate 0.0003 --num_epochs 50 --batch_size 1024 --num_workers 4 \
    --seed 3300 --save_weights \
    $EXTRA \
    --test_subject_ids "$S" --train_subject_ids $TRAIN \
    --output_dir "$R" --output_name "$NAME"
done
echo "DONE $ARM $*" >&2
