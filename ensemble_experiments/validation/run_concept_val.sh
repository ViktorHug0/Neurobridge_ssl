#!/bin/bash
# Regime 2: keep all nine training subjects, hold out a fixed 10% of training
# CONCEPTS for checkpoint selection.  Unlike LOSO-val this costs no subject, and
# --save_testsel_checkpoint gives a matched test-selected control from the same run.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <arm-name> <image-feature-dir>" >&2
  exit 2
fi

ARM=$1
IMAGE_FEATURE_DIR=$2
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
EEG_DATA_DIR=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
RUN_SEED=${RUN_SEED:-3300}
OUT="$REPO_ROOT/results/things_eeg/honest_ensemble/$ARM/seed$RUN_SEED"
ENCODER_TYPE=${ENCODER_TYPE:-TSConv_parameterizable}
FEATURE_DIM=${FEATURE_DIM:-512}
BACKBONE_DIM=${BACKBONE_DIM:-1024}
MIXUP_TYPE=${MIXUP_TYPE:-group}
TKERNEL=${TKERNEL:-25}
MAX_EPOCHS=${MAX_EPOCHS:-100}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-20}
VAL_CONCEPT_RATIO=${VAL_CONCEPT_RATIO:-0.10}

cd "$REPO_ROOT"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
test -f "$IMAGE_FEATURE_DIR/image_train.npy"
mkdir -p "$OUT"

for held in ${FOLDS:-1 2 3 4 5 6 7 8 9 10}; do
  tag=$(printf '%02d' "$held")
  if compgen -G "$OUT/*-sub-$tag/result.csv" >/dev/null; then
    echo "[skip] $ARM fold $tag already complete"
    continue
  fi
  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done
  echo "[$ARM] test=$held train=${train_ids[*]} (val = 10% of training concepts)"
  python train.py \
    --batch_size 1024 --num_workers "${NW:-6}" \
    --learning_rate 3e-4 --num_epochs "$MAX_EPOCHS" \
    --output_name "sub-$tag" --output_dir "$OUT" \
    --eeg_encoder_type "$ENCODER_TYPE" --tsconv_temporal_kernel "$TKERNEL" \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --select_best_on val --val_concept_ratio "$VAL_CONCEPT_RATIO" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" --early_stop_min_delta 0 \
    --save_testsel_checkpoint \
    --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$EEG_DATA_DIR" \
    --device cuda:0 --feature_dim "$FEATURE_DIM" --eeg_backbone_dim "$BACKBONE_DIM" \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --subject_mixup_mode raw_eeg --mixup_type "$MIXUP_TYPE" \
    --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
    --eval_mode plain_cosine --seed "$RUN_SEED"
done

count=$(find "$OUT" -mindepth 2 -maxdepth 2 -name result.csv | wc -l)
if [ "$count" -eq 10 ]; then
  python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
fi
