#!/bin/bash
# Run one of the six pre-registered decorrelated-twin arms over LOSO folds.
#
# Usage:
#   bash ensemble_experiments/decorrelated_models/run_first_wave.sh 0
#   SUBJECTS="1" bash ensemble_experiments/decorrelated_models/run_first_wave.sh control
#
# The Slurm array wrapper supplies indices 0..5.  The optional control is kept
# outside that array and is intentionally one fold by default.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

INDEX=${1:-${ARM_INDEX:-}}
if [ -z "$INDEX" ]; then
  echo "usage: $0 <0..5|control>" >&2
  exit 2
fi

case "$INDEX" in
  0) ARM=corr_l001_b0;  LAMBDA=0.01; BETA=0.0 ;;
  1) ARM=corr_l005_b0;  LAMBDA=0.05; BETA=0.0 ;;
  2) ARM=corr_l010_b0;  LAMBDA=0.10; BETA=0.0 ;;
  3) ARM=corr_l001_b05; LAMBDA=0.01; BETA=0.5 ;;
  4) ARM=corr_l005_b05; LAMBDA=0.05; BETA=0.5 ;;
  5) ARM=corr_l010_b05; LAMBDA=0.10; BETA=0.5 ;;
  control)
    ARM=corr_l000_b0_control
    LAMBDA=0.0
    BETA=0.0
    : "${SUBJECTS:=1}"
    ;;
  *) echo "unknown arm index: $INDEX (expected 0..5 or control)" >&2; exit 2 ;;
esac

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/first_wave}
DUMP_ROOT=${DUMP_ROOT:-$REPO_ROOT/results/things_eeg/synthetic_subjects/ensemble_screen/dumps}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}
ARM_ROOT="$RESULT_ROOT/$ARM/seed3300_3301"
mkdir -p "$ARM_ROOT" "$DUMP_ROOT"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "arm=$ARM lambda=$LAMBDA beta=$BETA subjects=$SUBJECTS"
  exit 0
fi

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[arm] $ARM lambda=$LAMBDA beta=$BETA subjects=$SUBJECTS"

active_lock=""
cleanup_lock() {
  if [ -n "$active_lock" ] && [ -d "$active_lock" ]; then
    rmdir "$active_lock"
  fi
}
trap cleanup_lock EXIT
trap 'cleanup_lock; exit 130' INT
trap 'cleanup_lock; exit 143' TERM

for held in $SUBJECTS; do
  tag=$(printf '%02d' "$held")
  if find "$ARM_ROOT" -mindepth 2 -maxdepth 2 -path "*-sub-$tag/result.csv" -print -quit | grep -q .; then
    echo "[skip] $ARM sub-$tag already complete"
    continue
  fi

  lock="$ARM_ROOT/.lock-sub-$tag"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "[lock] $ARM sub-$tag is already claimed"
    continue
  fi
  active_lock="$lock"

  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done
  fold_dir="$ARM_ROOT/$(date +%Y%m%d-%H%M%S)-sub-$tag"

  if python -m ensemble_experiments.decorrelated_models.train_twins \
      --arm "$ARM" \
      --output-dir "$fold_dir" \
      --dump-root "$DUMP_ROOT" \
      --held-subject "$held" \
      --train-subject-ids "${train_ids[@]}" \
      --eeg-data-dir "$EEG_DATA_DIR" \
      --image-feature-dir "$IMAGE_FEATURE_DIR" \
      --device "$DEVICE" \
      --num-workers "$NW" \
      --batch-size 1024 \
      --num-epochs 50 \
      --learning-rate 3e-4 \
      --weight-decay 1e-4 \
      --feature-dim 512 \
      --backbone-dim 1024 \
      --lambda-div "$LAMBDA" \
      --beta-ensemble "$BETA" \
      --decorrelation-start-epoch 11 \
      --seed-a 3300 \
      --seed-b 3301 \
      --train-rng-seed 7330 \
      --mixup-alpha 0.5; then
    rmdir "$lock"
    active_lock=""
  else
    status=$?
    rmdir "$lock"
    active_lock=""
    exit "$status"
  fi
done

completed=$(find "$ARM_ROOT" -mindepth 2 -maxdepth 2 -name result.csv | wc -l)
if [ "$completed" -eq 10 ]; then
  python compute_avg_results.py \
    --result_dir "$ARM_ROOT" \
    --output_name inter_subject_summary.csv
else
  echo "[partial] $ARM has $completed/10 completed folds"
fi
