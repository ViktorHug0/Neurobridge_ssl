#!/bin/bash
# Ten-fold ATM + TSConv score-decorrelation experiment at lambda=0.10.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

ARM=${ARM:-decor_atm_tsconv_l010_b0}
LAMBDA_DIV=${LAMBDA_DIV:-0.10}
BETA_ENSEMBLE=${BETA_ENSEMBLE:-0}
GAMMA_RESCUE=${GAMMA_RESCUE:-0}
RESCUE_TEMPERATURE=${RESCUE_TEMPERATURE:-0.5}
RESCUE_START_EPOCH=${RESCUE_START_EPOCH:-1}
SELECTION_PROTOCOL=${SELECTION_PROTOCOL:-test}
VAL_CONCEPT_RATIO=${VAL_CONCEPT_RATIO:-0.10}
VAL_CONCEPT_SEED=${VAL_CONCEPT_SEED:-20260822}
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/heterogeneous_wave}
DUMP_ROOT=${DUMP_ROOT:-$REPO_ROOT/results/things_eeg/synthetic_subjects/ensemble_screen/dumps}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-8}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_EPOCHS=${NUM_EPOCHS:-50}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}
ARM_ROOT="$RESULT_ROOT/$ARM/seed3300_3301"
mkdir -p "$ARM_ROOT" "$DUMP_ROOT"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[arm] $ARM selection=$SELECTION_PROTOCOL lambda=$LAMBDA_DIV beta=$BETA_ENSEMBLE gamma=$GAMMA_RESCUE rescue_tau=$RESCUE_TEMPERATURE rescue_start=$RESCUE_START_EPOCH batch=$BATCH_SIZE epochs=$NUM_EPOCHS ATM(128/128) + TSConv(1024/512) subjects=$SUBJECTS"

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

  python -m ensemble_experiments.decorrelated_models.train_twins \
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
    --encoder-a ATM \
    --backbone-dim-a 128 \
    --feature-dim-a 128 \
    --encoder-b TSConv_parameterizable \
    --backbone-dim-b 1024 \
    --feature-dim-b 512 \
    --lambda-div "$LAMBDA_DIV" \
    --beta-ensemble "$BETA_ENSEMBLE" \
    --decorrelation-start-epoch 11 \
    --gamma-rescue "$GAMMA_RESCUE" \
    --rescue-temperature "$RESCUE_TEMPERATURE" \
    --rescue-start-epoch "$RESCUE_START_EPOCH" \
    --selection-protocol "$SELECTION_PROTOCOL" \
    --val-concept-ratio "$VAL_CONCEPT_RATIO" \
    --val-concept-seed "$VAL_CONCEPT_SEED" \
    --seed-a 3300 \
    --seed-b 3301 \
    --train-rng-seed 7330 \
    --mixup-alpha 0.5
done

count=$(find "$ARM_ROOT" -mindepth 2 -maxdepth 2 -name result.csv | wc -l)
if [ "$count" -eq 10 ]; then
  flock "$ARM_ROOT/.aggregate.lock" python compute_avg_results.py \
    --result_dir "$ARM_ROOT" \
    --output_name inter_subject_summary.csv
fi
