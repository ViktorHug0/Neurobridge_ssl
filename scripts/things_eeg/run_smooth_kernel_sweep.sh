#!/bin/bash
# Does the smoothing arm lose because we under-smoothed? Sweep the strength over all subjects.
#
# The reference arm used NeuroBridge's upstream default (kernel 5, prob 0.3), which came out a
# wash over 10 subjects (-1.05, p=0.235). A one-subject probe at prob 1.0 then produced an
# isolated +3.5 at kernel 5 while kernels 1, 3 and 9 all landed exactly on the no-aug 36.50,
# which looks like a max-over-epochs artefact rather than an effect. This runs the full LOSO
# set so the question is settled on paired subjects instead of one run.
#
# Comparison points (seed 3300, paper commit, same base config):
#   results/things_eeg/subjectmix_rebuttal/smooth_aug_paperbase/  smoothing at the upstream default
#   tsconv_dropout_sweep_20260429-190741  SubjectMix 35.90 mean   _20260429-170207  no-aug 30.30
#
# RandomSmooth uses half = k//2 and window [t-half, t+half+1), so effective width is 2*(k//2)+1:
# k=1 is the identity, k=2 == k=3, k=4 == k=5, k=6 == k=7. Only odd k are worth running.
#
# Usage:  KERNELS="3 5 7" SUBJECTS="1 2 3 4 5 6 7 8 9 10" ./run_smooth_kernel_sweep.sh
set -eu
REPO=/nasbrain/p20fores/Neurobridge_SSL
# Must match run_smooth_paperbase.sh, which creates and patches this worktree.
WT=${SMOOTH_WORKTREE:-/nasbrain/p20fores/apr_code_3644141}
KERNELS=${KERNELS:-"3 5 7"}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}
P=${SMOOTH_P:-1.0}
SEED=${SEED:-3300}

[ -f "$WT/train.py" ] || { echo "FATAL: worktree missing; run run_smooth_paperbase.sh first" >&2; exit 1; }
grep -q "SMOOTH_K" "$WT/train.py" || { echo "FATAL: worktree train.py not parameterised" >&2; exit 1; }
( cd "$WT" && "$REPO/.venv/bin/python" test_random_smooth_vectorised.py >/dev/null ) \
  || { echo "FATAL: vectorised RandomSmooth does not match the reference loop" >&2; exit 1; }

cd "$WT"
for K in $KERNELS; do
  R=$REPO/results/things_eeg/subjectmix_rebuttal/smooth_kernel_sweep/seed${SEED}_k${K}_p${P}
  mkdir -p "$R"
  for S in $SUBJECTS; do
    NAME=$(printf 'sub-%02d' "$S")
    if compgen -G "$R/*-$NAME/result.csv" > /dev/null; then
      echo "== k=$K p=$P $NAME already done, skipping" >&2
      continue
    fi
    TRAIN=$(seq 1 10 | grep -vx "$S" | tr '\n' ' ')
    echo "== k=$K p=$P $NAME seed$SEED (train on $TRAIN)" >&2
    SMOOTH_K=$K SMOOTH_P=$P "$REPO/.venv/bin/python" train.py \
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
  echo "ARM DONE k=$K p=$P" >&2
done
echo "SWEEP DONE kernels=$KERNELS subjects=$SUBJECTS" >&2
