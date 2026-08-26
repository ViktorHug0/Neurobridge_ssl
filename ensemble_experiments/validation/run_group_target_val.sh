#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <arm-name> <image-feature-dir>" >&2
  exit 2
fi

ARM=$1
IMAGE_FEATURE_DIR=$2
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
EEG_DATA_DIR=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
OUT="$REPO_ROOT/results/things_eeg/honest_ensemble/$ARM/seed${RUN_SEED:-3300}"
ENCODER_TYPE=${ENCODER_TYPE:-TSConv_parameterizable}
FEATURE_DIM=${FEATURE_DIM:-512}
BACKBONE_DIM=${BACKBONE_DIM:-1024}
MIXUP_TYPE=${MIXUP_TYPE:-group}
MAX_EPOCHS=${MAX_EPOCHS:-100}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-20}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0}

encoder_args=(--eeg_encoder_type "$ENCODER_TYPE")
if [ "$ENCODER_TYPE" = TSConv_parameterizable ]; then
  encoder_args+=(
    --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50
  )
fi

cd "$REPO_ROOT"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
test -f "$IMAGE_FEATURE_DIR/image_train.npy"
test -f "$IMAGE_FEATURE_DIR/image_test.npy"
mkdir -p "$OUT"

# Default order deliberately includes an easy, hard, and middle fold. FOLDS can
# be overridden by a promotion job without changing the registered recipe.
for held in ${FOLDS:-1 3 5}; do
  tag=$(printf '%02d' "$held")
  val=$((held % 10 + 1))
  if compgen -G "$OUT/*-sub-$tag/result.csv" >/dev/null; then
    echo "[skip] $ARM outer fold $tag already complete"
    continue
  fi
  if ! mkdir "$OUT/.lock-$tag" 2>/dev/null; then
    echo "[lock] $ARM outer fold $tag claimed by another worker"
    continue
  fi

  train_ids=()
  for subject in $(seq 1 10); do
    if [ "$subject" -ne "$held" ] && [ "$subject" -ne "$val" ]; then
      train_ids+=("$subject")
    fi
  done

  echo "[$ARM] outer=$held val=$val train=${train_ids[*]}"
  python train.py \
    --batch_size 1024 --num_workers "${NW:-6}" \
    --learning_rate 3e-4 --num_epochs "$MAX_EPOCHS" \
    --output_name "sub-$tag" --output_dir "$OUT" \
    "${encoder_args[@]}" \
    --train_subject_ids "${train_ids[@]}" --val_subject_id "$val" \
    --test_subject_ids "$held" --select_best_on val \
    --early_stop_patience "$EARLY_STOP_PATIENCE" --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
    --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$EEG_DATA_DIR" \
    --device cuda:0 --feature_dim "$FEATURE_DIM" --eeg_backbone_dim "$BACKBONE_DIM" \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 8 --multi_positive_loss \
    --subject_mixup_mode raw_eeg --mixup_type "$MIXUP_TYPE" \
    --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
    --eval_mode plain_cosine --seed "${RUN_SEED:-3300}" --save_testsel_checkpoint
done

count=$(find "$OUT" -mindepth 2 -maxdepth 2 -name result.csv | wc -l)
if [ "$count" -eq 10 ]; then
  python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
fi
