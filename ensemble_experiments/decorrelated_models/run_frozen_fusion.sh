#!/bin/bash
# Train one EEG member against a fold-matched frozen ValCon reference member.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

: "${ARM:?ARM is required}"
: "${BETA_ENSEMBLE:?BETA_ENSEMBLE is required}"
: "${ENCODER_A:?ENCODER_A (the frozen encoder) is required}"
: "${BACKBONE_DIM_A:?BACKBONE_DIM_A is required}"
: "${FEATURE_DIM_A:?FEATURE_DIM_A is required}"
: "${FROZEN_CHECKPOINT_MEMBER:?member a or b in the reference run is required}"
: "${ENCODER_B:?ENCODER_B (the trainable encoder) is required}"
: "${BACKBONE_DIM_B:?BACKBONE_DIM_B is required}"
: "${FEATURE_DIM_B:?FEATURE_DIM_B is required}"

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/frozen_fusion_wave}
REFERENCE_ROOT=${REFERENCE_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/valcon_wave/rescue_valcon_atm_tsconv_g000_b0_bs512/seed3300_3301}
DUMP_ROOT=${DUMP_ROOT:-$REPO_ROOT/results/things_eeg/synthetic_subjects/ensemble_screen/dumps}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6"}
SEED_A=${SEED_A:-3300}
SEED_B=${SEED_B:-3301}
TRAIN_RNG_SEED=${TRAIN_RNG_SEED:-7330}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_EPOCHS=${NUM_EPOCHS:-50}
ARM_ROOT="$RESULT_ROOT/$ARM/seed${SEED_A}_${SEED_B}"
mkdir -p "$ARM_ROOT" "$DUMP_ROOT"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[frozen-fusion] arm=$ARM beta=$BETA_ENSEMBLE frozen=$ENCODER_A trainable=$ENCODER_B subjects=$SUBJECTS"

for held in $SUBJECTS; do
  tag=$(printf '%02d' "$held")
  if find "$ARM_ROOT" -mindepth 2 -maxdepth 2 -path "*-sub-$tag/result.csv" -print -quit | grep -q .; then
    echo "[skip] $ARM sub-$tag already complete"
    continue
  fi

  mapfile -t frozen_candidates < <(
    find "$REFERENCE_ROOT" -mindepth 2 -maxdepth 2 \
      -path "*-sub-$tag/checkpoint_member_${FROZEN_CHECKPOINT_MEMBER}_val_best.pth" \
      -type f | sort
  )
  if [ "${#frozen_candidates[@]}" -ne 1 ]; then
    echo "expected exactly one frozen checkpoint for sub-$tag; found ${#frozen_candidates[@]}" >&2
    exit 2
  fi
  frozen_checkpoint=${frozen_candidates[0]}

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
    --encoder-a "$ENCODER_A" \
    --backbone-dim-a "$BACKBONE_DIM_A" \
    --feature-dim-a "$FEATURE_DIM_A" \
    --encoder-b "$ENCODER_B" \
    --backbone-dim-b "$BACKBONE_DIM_B" \
    --feature-dim-b "$FEATURE_DIM_B" \
    --lambda-div 0 \
    --beta-ensemble "$BETA_ENSEMBLE" \
    --gamma-rescue 0 \
    --selection-protocol valcon \
    --val-concept-ratio 0.10 \
    --val-concept-seed 20260822 \
    --freeze-member a \
    --frozen-checkpoint "$frozen_checkpoint" \
    --fusion-loss-mode deployed_unique \
    --seed-a "$SEED_A" \
    --seed-b "$SEED_B" \
    --train-rng-seed "$TRAIN_RNG_SEED" \
    --mixup-alpha 0.5
done
