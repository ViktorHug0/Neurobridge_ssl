#!/bin/bash
# Ten-fold ATM + TSConv score-decorrelation experiment.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

ARM=decor_atm_tsconv_l005_b0
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/heterogeneous_wave}
DUMP_ROOT=${DUMP_ROOT:-$REPO_ROOT/results/things_eeg/synthetic_subjects/ensemble_screen/dumps}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}
ARM_ROOT="$RESULT_ROOT/$ARM/seed3300_3301"
mkdir -p "$ARM_ROOT" "$DUMP_ROOT"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[arm] $ARM lambda=0.05 beta=0 ATM(128/128) + TSConv(1024/512) subjects=$SUBJECTS"

for held in $SUBJECTS; do
  tag=$(printf '%02d' "$held")
  if find "$ARM_ROOT" -mindepth 2 -maxdepth 2 -path "*-sub-$tag/result.csv" -print -quit | grep -q .; then
    echo "[skip] $ARM sub-$tag already complete"
    continue
  fi

  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done
  fold_dir="$ARM_ROOT/$(date +%Y%m%d-%H%M%S)-sub-$tag"

  python -m ensemble_experiments.decorrelated_models.train_twins     --arm "$ARM"     --output-dir "$fold_dir"     --dump-root "$DUMP_ROOT"     --held-subject "$held"     --train-subject-ids "${train_ids[@]}"     --eeg-data-dir "$EEG_DATA_DIR"     --image-feature-dir "$IMAGE_FEATURE_DIR"     --device "$DEVICE"     --num-workers "$NW"     --batch-size 1024     --num-epochs 50     --learning-rate 3e-4     --weight-decay 1e-4     --encoder-a ATM     --backbone-dim-a 128     --feature-dim-a 128     --encoder-b TSConv_parameterizable     --backbone-dim-b 1024     --feature-dim-b 512     --lambda-div 0.05     --beta-ensemble 0     --decorrelation-start-epoch 11     --seed-a 3300     --seed-b 3301     --train-rng-seed 7330     --mixup-alpha 0.5
done

python compute_avg_results.py   --result_dir "$ARM_ROOT"   --output_name inter_subject_summary.csv
