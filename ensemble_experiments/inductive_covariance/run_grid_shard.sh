#!/bin/bash
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

: "${SUBJECTS:?SUBJECTS is required, e.g. '3 4'}"

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/inductive_covariance/single_tsconv_screen}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
EPOCHS=${EPOCHS:-50}
REF_TRIALS=${REF_TRIALS:-2048}

# Put the mental-imagery-centred setting first so all four subjects acquire a
# directly comparable result early. The remaining entries complete the 3x3.
COMBINATIONS=(
  "0.40 0.25"
  "0.20 0.25"
  "0.60 0.25"
  "0.40 0.10"
  "0.40 0.50"
  "0.20 0.10"
  "0.20 0.50"
  "0.60 0.10"
  "0.60 0.50"
)

# COMBOS overrides the default 3x3 grid: "alpha shrinkage" pairs, ';'-separated.
if [ -n "${COMBOS:-}" ]; then
  IFS=';' read -ra COMBINATIONS <<< "$COMBOS"
fi

CACHE_DIR=${CACHE_DIR:-$RESULT_ROOT/cache}
mkdir -p "$CACHE_DIR" "$RESULT_ROOT/local_runner"
echo "[cov-grid] host=$(hostname) subjects=$SUBJECTS epochs=$EPOCHS device=$DEVICE"

for combination in "${COMBINATIONS[@]}"; do
  read -r alpha shrinkage <<< "$combination"
  alpha_tag=${alpha/./}
  shrinkage_tag=${shrinkage/./}
  variant="alpha${alpha_tag}_shrink${shrinkage_tag}"
  for target in $SUBJECTS; do
    target_tag=$(printf '%02d' "$target")
    train_ids=()
    for subject in {1..10}; do
      if [ "$subject" -ne "$target" ]; then
        train_ids+=("$subject")
      fi
    done
    output_dir="$RESULT_ROOT/$variant"
    reference_cache="$CACHE_DIR/source_cov_sub-${target_tag}.pt"
    echo "[$(date -Is)] variant=$variant target=$target train=${train_ids[*]}"
    python train.py \
      --batch_size 1024 \
      --num_workers "$NW" \
      --learning_rate 3e-4 \
      --output_name "sub-${target_tag}" \
      --eeg_encoder_type TSConv \
      --train_subject_ids "${train_ids[@]}" \
      --test_subject_ids "$target" \
      --select_best_on test \
      --softplus \
      --num_epochs "$EPOCHS" \
      --image_feature_dir "$IMAGE_FEATURE_DIR" \
      --text_feature_dir "" \
      --eeg_data_dir "$EEG_DATA_DIR" \
      --device "$DEVICE" \
      --output_dir "$output_dir" \
      --img_l2norm \
      --projector linear \
      --feature_dim 512 \
      --eeg_backbone_dim 1024 \
      --data_average \
      --seed 3300 \
      --multi_positive_loss \
      --grouped_batch_sampler \
      --samples_per_image 9 \
      --subject_mixup_mode raw_eeg \
      --mixup_type pairwise \
      --subject_mixup_alpha 0.5 \
      --inductive_covariance_align \
      --inductive_covariance_alpha "$alpha" \
      --inductive_covariance_shrinkage "$shrinkage" \
      --inductive_covariance_ref_trials "$REF_TRIALS" \
      --inductive_covariance_reference_cache "$reference_cache"

    flock "$RESULT_ROOT/.summary.lock" \
      python ensemble_experiments/inductive_covariance/summarize_grid.py \
        --result-root "$RESULT_ROOT"
  done
done

echo "[$(date -Is)] shard complete subjects=$SUBJECTS"
