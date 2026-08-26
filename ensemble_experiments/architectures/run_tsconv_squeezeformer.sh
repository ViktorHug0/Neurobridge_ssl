#!/bin/bash
# Gate TSConv-Squeezeformer on subjects 1-5, then finish LOSO only if competitive.
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

IMAGE_FEATURE_DIR="$PWD/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
OUTPUT_DIR="$PWD/results/things_eeg/ensemble_arch_candidates/tsconv_squeezeformer_pairwise50/seed3300"

test -d "$IMAGE_FEATURE_DIR"
test -d "$EEG_DATA_DIR"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "$OUTPUT_DIR"

run_fold() {
  local held="$1"
  local tag
  tag=$(printf '%02d' "$held")
  if compgen -G "$OUTPUT_DIR/*-sub-$tag/result.csv" > /dev/null; then
    echo "[skip] TSConv-Squeezeformer fold $tag already complete"
    return
  fi

  local train_ids=()
  local subject
  for subject in $(seq 1 10); do
    if [ "$subject" -ne "$held" ]; then
      train_ids+=("$subject")
    fi
  done

  echo "[run] TSConv-Squeezeformer fold $tag"
  python train.py \
    --batch_size 1024 --num_workers "${NW:-6}" \
    --learning_rate 3e-4 --num_epochs 50 \
    --output_name "sub-$tag" --output_dir "$OUTPUT_DIR" \
    --eeg_encoder_type OrthoTSConvSqueezeformer \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --image_feature_dir "$IMAGE_FEATURE_DIR" \
    --eeg_data_dir "$EEG_DATA_DIR" \
    --device cuda:0 --feature_dim 512 --eeg_backbone_dim 1024 \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --subject_mixup_mode raw_eeg --mixup_type pairwise \
    --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
    --eval_mode plain_cosine --select_best_on test --seed 3300
}

for held in $(seq 1 5); do
  run_fold "$held"
done

python compute_avg_results.py \
  --result_dir "$OUTPUT_DIR" \
  --output_name first_five_summary.csv

if ! python ensemble_experiments/architectures/strength_gate.py \
  "$OUTPUT_DIR" --subjects 1 2 3 4 5 --reference 36.60 --max-gap 5.0; then
  echo "[drop] TSConv-Squeezeformer missed the five-subject strength gate; stopping."
  exit 0
fi

for held in $(seq 6 10); do
  run_fold "$held"
done

python compute_avg_results.py \
  --result_dir "$OUTPUT_DIR" \
  --output_name inter_subject_summary.csv
