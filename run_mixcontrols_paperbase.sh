#!/bin/bash
# Controls isolating WHAT in SubjectMix produces the gain (R3 Q2), at the paper's commit 3644141.
#
# SubjectMix mixes two trials of the SAME stimulus from DIFFERENT subjects, on the raw EEG, and
# leaves the image target untouched. Two properties could each explain the gain, and each arm
# below removes exactly one of them:
#
#   cross_stimulus_mix  partner comes from a different stimulus. Target and positive mask
#                       unchanged, so only the same-stimulus constraint is removed.
#   noise_matched       isotropic Gaussian noise whose per-sample norm equals the perturbation
#                       SubjectMix would have applied to that sample. Magnitude held fixed,
#                       direction randomised, so only "toward another subject" is removed.
#   global_mixup        ordinary mixup: global permutation, both EEG and image mixed. Kept for
#                       reference; it changes the objective as well, so it is the blunter probe.
#
# Runs from a SEPARATE worktree carrying these patches, so the other paper-commit arms run
# against unmodified 3644141. Self-checks: test_controls.py, test_global_mixup.py.
#
# Usage:  run_mixcontrols_paperbase.sh <arm> <subject> [<subject> ...]
set -eu
REPO=/nasbrain/p20fores/Neurobridge_SSL
WT=${MIXUP_WORKTREE:-/nasbrain/p20fores/apr_code_mixup}
SEED=3300

ARM=$1; shift
case "$ARM" in
  cross_stimulus_mix) FLAGS="--cross_stimulus_mix --subject_mixup_alpha 0.5" ;;
  noise_matched)      FLAGS="--noise_matched --subject_mixup_alpha 0.5" ;;
  mixco)              FLAGS="--mixco --subject_mixup_alpha 0.5" ;;
  global_mixup)       FLAGS="--global_mixup --subject_mixup_alpha 0.5" ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

R=$REPO/results/things_eeg/subjectmix_rebuttal/${ARM}_paperbase/seed$SEED
mkdir -p "$R"
cd "$WT"
"$REPO/.venv/bin/python" test_controls.py || { echo "FATAL: control self-check failed" >&2; exit 1; }
"$REPO/.venv/bin/python" test_mixco.py     || { echo "FATAL: mixco self-check failed" >&2; exit 1; }

for S in "$@"; do
  NAME=$(printf 'sub-%02d' "$S")
  if compgen -G "$R/*-$NAME/result.csv" > /dev/null; then
    echo "== $ARM $NAME already done, skipping" >&2
    continue
  fi
  TRAIN=$(seq 1 10 | grep -vx "$S" | tr '\n' ' ')
  echo "== $ARM $NAME (train on $TRAIN)" >&2
  # identical to the published SubjectMix arm except the arm flag replaces
  # --subject_mixup_mode raw_eeg; subject mixup stays off so the two cannot stack
  "$REPO/.venv/bin/python" train.py \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
    --image_feature_dir "$REPO/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit" \
    --text_feature_dir '' \
    --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
    --eeg_backbone_dim 1024 --projector linear --softplus --img_l2norm \
    --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --data_average \
    $FLAGS \
    --learning_rate 0.0003 --num_epochs 50 --batch_size 1024 --num_workers 4 \
    --seed "$SEED" --save_weights \
    --test_subject_ids "$S" --train_subject_ids $TRAIN \
    --output_dir "$R" --output_name "$NAME"
done
echo "DONE $ARM $*" >&2
