#!/bin/bash
# Width probe: same arms, FD/BB overridable. ATM loses ~5 points at bb1024 vs
# bb128, so the TSConv width may be handicapping the ortho encoders too.
#
# Recipe is held IDENTICAL to tsconv_iv (the mainline ensemble member): InternViT
# layer 28, pairwise SubjectMix a=0.5 raw_eeg, 50 epochs, fd512/bb1024, seed 3300.
# Only --eeg_encoder_type varies, so any accuracy difference is the architecture.
#
# Kill-fast gates (baseline averages 40.17 over folds 1-3 and 35.90 over folds 1-5):
#   after fold 3 -> floor 31.0   (the >30% target, scaled to the easy early folds)
#   after fold 5 -> floor 28.0   (repo convention)
set -euo pipefail
ARM=${1:?usage: run_ortho.sh <arm>}
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

case "$ARM" in
  riemann) ENC=OrthoRiemann ;;
  sincpow) ENC=OrthoSincPow ;;
  spec)    ENC=OrthoSpec ;;
  mixer)   ENC=OrthoMixer ;;
  tcn)     ENC=OrthoTCN ;;
  convgru) ENC=OrthoConvGRU ;;
  covpool) ENC=OrthoCovPool ;;
  perceiver) ENC=OrthoPerceiver ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

OUT=$PWD/results/things_eeg/ortho_arch/${ARM}_bb${BB:-1024}/seed3300
mkdir -p "$OUT"
# FOLDS lets a second worker backfill a disjoint subset of folds on another GPU.
for held in ${FOLDS:-$(seq 1 10)}; do
  tag=$(printf '%02d' "$held")
  if compgen -G "$OUT/*-sub-$tag/result.csv" > /dev/null; then
    echo "[skip] fold $tag already done"; continue
  fi
  if ! mkdir "$OUT/.lock-$tag" 2>/dev/null; then
    echo "[lock] fold $tag claimed by another worker"; continue
  fi
  train_ids=()
  for subject in $(seq 1 10); do [ "$subject" -ne "$held" ] && train_ids+=("$subject"); done
  python train.py \
    --batch_size 1024 --num_workers ${NW:-8} --learning_rate 3e-4 --num_epochs 50 \
    --output_name "sub-$tag" --output_dir "$OUT" \
    --eeg_encoder_type "$ENC" \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --image_feature_dir "$PWD/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
    --device cuda:0 --feature_dim ${FD:-512} --eeg_backbone_dim ${BB:-1024} \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --mixup_type pairwise --subject_mixup_alpha 0.5 --subject_mixup_mode raw_eeg \
    --eval_mode plain_cosine --select_best_on test --seed 3300

  if [ -z "${FOLDS:-}" ]; then
    if [ "$held" -eq 3 ]; then
      python ensemble_experiments/architectures/abort_check.py "$OUT" 31.0 \
        || { echo "ABORTED $ARM at fold 3" >&2; exit 3; }
    elif [ "$held" -eq 5 ]; then
      python ensemble_experiments/architectures/abort_check.py "$OUT" 28.0 \
        || { echo "ABORTED $ARM at fold 5" >&2; exit 3; }
    fi
  fi
done
if [ "$(ls -d "$OUT"/*-sub-*/result.csv 2>/dev/null | wc -l)" -eq 10 ]; then
  python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
fi
