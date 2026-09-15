#!/bin/bash
# Signal-smoothing augmentation on the paper's exact base config, at the paper's commit.
#
# Why a worktree: the published SubjectMix / no-SubjectMix runs
#   results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741  (SubjectMix, 35.90)
#   results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-170207  (baseline,   30.30)
# were produced by commit 3644141. Today's HEAD gives 34.80 instead of 35.90 on the same
# config and seed -- same flags, drifted code -- so a smoothing arm run at HEAD would not be
# comparable to those numbers. Verified: 3644141 reproduces sub-01 = 50.50 exactly.
#
# module/training_plots.py is imported by train.py at that commit but was never committed
# (it landed later in a14485b), so the worktree needs it copied in. It only saves matplotlib
# figures and cannot affect results.
#
# Usage:  run_smooth_paperbase.sh <seed> <subject> [<subject> ...]
#   e.g.  run_smooth_paperbase.sh 3300 1 2 3 4 5 6 7 8 9 10
set -eu
REPO=/nasbrain/p20fores/Neurobridge_SSL
# Stable sibling worktree already checked out at $COMMIT. The previous default pointed into a
# per-session /tmp scratch path that no longer exists, so every run rebuilt the worktree (or
# failed outright). apr_code_aug is taken by resswap.sbatch; apr_code_mixup by the mixup driver.
WT=${SMOOTH_WORKTREE:-/nasbrain/p20fores/apr_code_3644141}
COMMIT=3644141

if [ ! -f "$WT/train.py" ]; then
  echo "creating worktree at $WT ($COMMIT)" >&2
  git -C "$REPO" worktree add "$WT" "$COMMIT"
fi
cp -n "$REPO/module/training_plots.py" "$WT/module/" 2>/dev/null || true
# RandomSmooth at this commit is a per-channel/per-timepoint Python loop costing 74 ms a
# trial, which makes one subject take ~4 h of pure augmentation. The replacement is the
# same truncated moving average, vectorised, and consumes the RNG identically (verified by
# the test next to it), so results are unchanged and a subject takes ~25 min instead.
cp "$REPO/scripts/things_eeg/patches/eeg_augmentation_fast_smooth.py" "$WT/module/eeg_augmentation.py"
cp "$REPO/scripts/things_eeg/patches/test_random_smooth_vectorised.py" "$WT/"
( cd "$WT" && "$REPO/.venv/bin/python" test_random_smooth_vectorised.py >/dev/null ) \
  || { echo "FATAL: vectorised RandomSmooth does not match the reference loop" >&2; exit 1; }

SEED=$1; shift
R=$REPO/results/things_eeg/subjectmix_rebuttal/smooth_aug_paperbase/seed$SEED
mkdir -p "$R"
cd "$WT"

for S in "$@"; do
  NAME=$(printf 'sub-%02d' "$S")
  if compgen -G "$R/*-$NAME/result.csv" > /dev/null; then
    echo "== smooth seed$SEED $NAME already done, skipping" >&2
    continue
  fi
  TRAIN=$(seq 1 10 | grep -vx "$S" | tr '\n' ' ')
  echo "== smooth seed$SEED $NAME (train on $TRAIN)" >&2
  # identical to the published SubjectMix arm except --eeg_aug --eeg_aug_type smooth
  # replaces --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5
  "$REPO/.venv/bin/python" train.py \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
    --image_feature_dir "$REPO/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit" \
    --text_feature_dir '' \
    --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
    --eeg_backbone_dim 1024 --projector linear --softplus --img_l2norm \
    --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --data_average \
    --eeg_aug --eeg_aug_type smooth \
    --learning_rate 0.0003 --num_epochs 50 --batch_size 1024 --num_workers 4 \
    --seed "$SEED" --save_weights \
    --test_subject_ids "$S" --train_subject_ids $TRAIN \
    --output_dir "$R" --output_name "$NAME"
done
echo "DONE smooth seed$SEED $*" >&2
