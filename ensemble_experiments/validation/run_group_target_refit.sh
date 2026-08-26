#!/bin/bash
# Refit one validation-tuned arm on all nine source subjects for the selected
# epoch count.  The final epoch is retained unconditionally (`fixed` mode), so
# no held-out-subject metric can affect the checkpoint.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <arm-name> <image-feature-dir> <outer-fold>" >&2
  exit 2
fi

ARM=$1
IMAGE_FEATURE_DIR=$2
HELD=$3
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
EEG_DATA_DIR=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
TUNE_ROOT=${TUNE_ROOT_OVERRIDE:-"$REPO_ROOT/results/things_eeg/honest_ensemble/$ARM/seed3300"}
OUT=${OUT_OVERRIDE:-"$REPO_ROOT/results/things_eeg/honest_ensemble/${ARM}_refit9/seed3300"}
TAG=$(printf '%02d' "$HELD")
ENCODER_TYPE=${ENCODER_TYPE:-TSConv_parameterizable}
FEATURE_DIM=${FEATURE_DIM:-512}
BACKBONE_DIM=${BACKBONE_DIM:-1024}
MIXUP_TYPE=${MIXUP_TYPE:-group}
RUN_SEED=${RUN_SEED:-3300}

encoder_args=(--eeg_encoder_type "$ENCODER_TYPE")
if [ "$ENCODER_TYPE" = TSConv_parameterizable ]; then
  encoder_args+=(
    --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50
  )
fi

cd "$REPO_ROOT"
source .venv/bin/activate

TUNE_RESULT=$(find "$TUNE_ROOT" -maxdepth 2 -path "*-sub-$TAG/result.csv" -print -quit)
if [ -z "$TUNE_RESULT" ]; then
  echo "missing validation-tuning result for $ARM outer=$HELD" >&2
  exit 3
fi
EPOCHS=$(python - "$TUNE_RESULT" <<'PY'
import pandas as pd
import sys

value = int(round(float(pd.read_csv(sys.argv[1]).iloc[0]["best epoch"])))
if value <= 0:
    raise SystemExit(f"invalid selected epoch: {value}")
print(value)
PY
)

if compgen -G "$OUT/*-sub-$TAG/result.csv" >/dev/null; then
  echo "[skip] ${ARM}_refit9 outer=$HELD already complete"
  exit 0
fi
mkdir -p "$OUT"
train_ids=()
for subject in $(seq 1 10); do
  [ "$subject" -ne "$HELD" ] && train_ids+=("$subject")
done

echo "[${ARM}_refit9] outer=$HELD fixed_epochs=$EPOCHS train=${train_ids[*]}"
python train.py \
  --batch_size 1024 --num_workers "${NW:-6}" \
  --learning_rate 3e-4 --num_epochs "$EPOCHS" \
  --output_name "sub-$TAG" --output_dir "$OUT" \
  "${encoder_args[@]}" \
  --train_subject_ids "${train_ids[@]}" --test_subject_ids "$HELD" \
  --select_best_on fixed \
  --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$EEG_DATA_DIR" \
  --device cuda:0 --feature_dim "$FEATURE_DIM" --eeg_backbone_dim "$BACKBONE_DIM" \
  --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
  --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
  --subject_mixup_mode raw_eeg --mixup_type "$MIXUP_TYPE" \
  --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
  --eval_mode plain_cosine --seed "$RUN_SEED"
