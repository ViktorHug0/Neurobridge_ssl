#!/bin/bash
# Full ten-fold test-selected run for one EEG architecture / image-target pair.
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

: "${ARM:?ARM must name the output arm}"
: "${ENCODER:?ENCODER must name the registered EEG encoder}"
: "${IMAGE_FEATURE_DIR:?IMAGE_FEATURE_DIR must name the frozen target directory}"
EEG_DATA_DIR=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
OUTPUT_DIR="$PWD/results/things_eeg/ensemble_arch_candidates/target_transfer_testsel/$ARM/seed3300"

test -f "$IMAGE_FEATURE_DIR/image_train.npy"
test -d "$EEG_DATA_DIR"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "$OUTPUT_DIR"

for held in ${FOLDS:-1 2 3 4 5 6 7 8 9 10}; do
  tag=$(printf '%02d' "$held")
  if compgen -G "$OUTPUT_DIR/*-sub-$tag/result.csv" > /dev/null; then
    echo "[skip] $ARM fold $tag already complete"
    continue
  fi

  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done

  echo "[run] $ARM fold $tag"
  python train.py \
    --batch_size 1024 --num_workers "${NW:-6}" \
    --learning_rate 3e-4 --num_epochs 50 \
    --output_name "sub-$tag" --output_dir "$OUTPUT_DIR" \
    --eeg_encoder_type "$ENCODER" \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$EEG_DATA_DIR" \
    --device cuda:0 --feature_dim 512 --eeg_backbone_dim 1024 \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --subject_mixup_mode raw_eeg --mixup_type pairwise \
    --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
    --eval_mode plain_cosine --select_best_on test --seed 3300
done

count=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -name result.csv | wc -l)
if [ "$count" -eq 10 ]; then
  python compute_avg_results.py \
    --result_dir "$OUTPUT_DIR" --output_name inter_subject_summary.csv
  python ensemble_experiments/test_selected/dump_testselected_arm.py \
    --arm "$ARM" --run-root "$OUTPUT_DIR" \
    --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
else
  echo "[partial] $ARM has $count/10 folds"
fi
