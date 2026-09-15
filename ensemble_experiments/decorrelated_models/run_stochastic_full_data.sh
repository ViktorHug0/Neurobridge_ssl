#!/bin/bash
# Matched full-source joint-training control and MI-inspired stochastic twin arm.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

VARIANT=${1:-}
case "$VARIANT" in
  control)
    ARM=joint_b05_control
    SPEC_SD=0.0
    CHANNEL_DROP=0.0
    MEMBER_KEEP=1.0
    ;;
  stochastic)
    ARM=stoch_spec05_cdrop10_keep75_b05
    SPEC_SD=0.5
    CHANNEL_DROP=0.1
    MEMBER_KEEP=0.75
    ;;
  *) echo "usage: $0 <control|stochastic>" >&2; exit 2 ;;
esac

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/stochastic_full_data_wave}
DUMP_ROOT=${DUMP_ROOT:-$RESULT_ROOT/dumps}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}
NUM_EPOCHS=${NUM_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-512}
ARM_ROOT="$RESULT_ROOT/$ARM/seed3300_3301"
mkdir -p "$ARM_ROOT" "$DUMP_ROOT"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "arm=$ARM subjects=$SUBJECTS spec_sd=$SPEC_SD channel_drop=$CHANNEL_DROP member_keep=$MEMBER_KEEP"
  exit 0
fi

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[stochastic-full-data] arm=$ARM subjects=$SUBJECTS epochs=$NUM_EPOCHS"

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
      --batch-size "$BATCH_SIZE" \
      --num-epochs "$NUM_EPOCHS" \
      --learning-rate 3e-4 \
      --weight-decay 1e-4 \
      --feature-dim 512 \
      --backbone-dim 1024 \
      --lambda-div 0 \
      --beta-ensemble 0.5 \
      --fusion-loss-mode deployed_unique \
      --member-spectral-gain-sd "$SPEC_SD" \
      --member-spectral-control-points 8 \
      --member-channel-drop-prob "$CHANNEL_DROP" \
      --member-keep-prob "$MEMBER_KEEP" \
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

python -m ensemble_experiments.decorrelated_models.analyze_stochastic_full_data \
  --result-root "$RESULT_ROOT"
