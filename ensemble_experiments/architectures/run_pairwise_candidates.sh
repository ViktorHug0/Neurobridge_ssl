#!/bin/bash
# Full LOSO comparison of the temporal Conformer and 1-D ConvNeXt candidates.
# Both use the test-selected, pairwise-SubjectMix/50-epoch family-A recipe.
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

IMAGE_FEATURE_DIR="$PWD/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
RESULT_ROOT="$PWD/results/things_eeg/ensemble_arch_candidates"

test -d "$IMAGE_FEATURE_DIR"
test -d "$EEG_DATA_DIR"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

for candidate in \
  "temporal_conformer_pairwise50:OrthoTemporalConformer" \
  "convnext1d_pairwise50:OrthoConvNeXt1D"
do
  IFS=: read -r arm encoder <<< "$candidate"
  output_dir="$RESULT_ROOT/$arm/seed3300"
  mkdir -p "$output_dir"

  for held in $(seq 1 10); do
    tag=$(printf '%02d' "$held")
    if compgen -G "$output_dir/*-sub-$tag/result.csv" > /dev/null; then
      echo "[skip] $arm fold $tag already complete"
      continue
    fi

    train_ids=()
    for subject in $(seq 1 10); do
      if [ "$subject" -ne "$held" ]; then
        train_ids+=("$subject")
      fi
    done

    echo "[run] $arm fold $tag"
    python train.py \
      --batch_size 1024 --num_workers "${NW:-6}" \
      --learning_rate 3e-4 --num_epochs 50 \
      --output_name "sub-$tag" --output_dir "$output_dir" \
      --eeg_encoder_type "$encoder" \
      --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
      --image_feature_dir "$IMAGE_FEATURE_DIR" \
      --eeg_data_dir "$EEG_DATA_DIR" \
      --device cuda:0 --feature_dim 512 --eeg_backbone_dim 1024 \
      --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
      --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
      --subject_mixup_mode raw_eeg --mixup_type pairwise \
      --subject_mixup_alpha 0.5 --subject_mixup_prob 1.0 \
      --eval_mode plain_cosine --select_best_on test --seed 3300
  done

  python compute_avg_results.py \
    --result_dir "$output_dir" \
    --output_name inter_subject_summary.csv
done
