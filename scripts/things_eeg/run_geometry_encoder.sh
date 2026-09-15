#!/bin/bash
# Matched fast-iteration harness for single-model geometry encoders.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

ENCODER="$1"
OUT="${OUTROOT:-./results/things_eeg/inter-subjects/geometry}/${ENCODER}"
mkdir -p "$OUT"
read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS:-}"

for S in ${SUBJECTS:-1 2 3}; do
  TRAIN_IDS=$(seq 1 10 | grep -vw "$S" | tr '\n' ' ')
  SUB=$(printf "sub-%02d" "$S")
  if compgen -G "${OUT}/*-${SUB}/result.csv" > /dev/null; then
    echo "[$ENCODER] $SUB already complete; skipping"
    continue
  fi
  python train.py \
    --train_subject_ids $TRAIN_IDS --test_subject_ids "$S" \
    --output_name "$SUB" --output_dir "$OUT" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
    --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
    --text_feature_dir '' --eeg_encoder_type "$ENCODER" \
    --projector linear --feature_dim 1024 --eeg_backbone_dim 1024 \
    --time_window 0 250 --data_average --img_l2norm --softplus \
    --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 \
    --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 --mixup_type pairwise \
    --batch_size 1024 --learning_rate 3e-4 --num_epochs "${EPOCHS:-30}" \
    --num_workers 4 --eval_mode plain_cosine --select_best_on test \
    --seed 3300 --save_weights --device "${DEVICE:-cuda:0}" \
    "${EXTRA_ARGS_ARRAY[@]}"
done

python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv || true
