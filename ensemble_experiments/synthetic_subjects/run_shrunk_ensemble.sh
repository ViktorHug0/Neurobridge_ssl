#!/bin/bash
# Shrunk-width rerun of the winning k=3 cross-backbone ensemble.
#
# Every arm keeps the exact recipe shared by the three original winners
# (50 epochs, batch 1024, lr 3e-4, pairwise SubjectMix a=0.5 on raw_eeg,
# samples_per_image 9, grouped batch sampler, multi-positive loss) and changes
# ONLY the width: eeg_backbone_dim 1024 -> 128, feature_dim 512 -> 128.
# ATM (atm_iv) is not retrained: it is already bb=128/fd=128 on this recipe.
#
# usage: run_shrunk_ensemble.sh <arm>
set -euo pipefail

ARM=${1:?usage: run_shrunk_ensemble.sh <arm>}

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

SEED=3300
case "$ARM" in
  eva_s3300)  IMG=EVA02-E-14_layer35_mean ;;
  vith_s3300) IMG=ViT-H-14_layer10_mean ;;
  eva_s3301)  IMG=EVA02-E-14_layer35_mean; SEED=3301 ;;
  eva_s3302)  IMG=EVA02-E-14_layer35_mean; SEED=3302 ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

OUT=$PWD/results/things_eeg/synthetic_subjects/shrunk_ensemble/$ARM/seed$SEED
mkdir -p "$OUT"

for held in $(seq 1 10); do
  tag=$(printf '%02d' "$held")
  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done
  python train.py \
    --batch_size 1024 --num_workers 8 --learning_rate 3e-4 --num_epochs 50 \
    --output_name "sub-$tag" --output_dir "$OUT" \
    --eeg_encoder_type TSConv \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --image_feature_dir "$PWD/data/things_eeg/image_feature/$IMG" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
    --device cuda:0 --feature_dim 128 --eeg_backbone_dim 128 \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --mixup_type pairwise --subject_mixup_alpha 0.5 --subject_mixup_mode raw_eeg \
    --eval_mode plain_cosine --select_best_on test --seed "$SEED"
done

python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
