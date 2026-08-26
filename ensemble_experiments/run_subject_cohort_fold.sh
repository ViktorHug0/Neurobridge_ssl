#!/bin/bash
# Train and export one TSConv model for one three-subject source cohort.
set -euo pipefail

: "${TARGET:?TARGET is required}"
: "${COHORT:?COHORT is required}"

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/subject_cohort_bagging/testselected_triplets}
GROUP_MODE=${GROUP_MODE:-triplet}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}

target_tag=$(printf '%02d' "$TARGET")
fold_root="$RESULT_ROOT/target-$target_tag/cohort-$COHORT"
mkdir -p "$fold_root"
read -ra train_ids <<< "$(
  python -m ensemble_experiments.analyze_subject_cohort_bagging \
    --group-mode "$GROUP_MODE" \
    --print-group "$TARGET" "$COHORT"
)"

echo "[cohort] mode=$GROUP_MODE target=$TARGET cohort=$COHORT train=${train_ids[*]}"
python train.py \
  --batch_size 1024 \
  --num_workers "$NW" \
  --learning_rate 3e-4 \
  --output_name model \
  --eeg_encoder_type TSConv \
  --train_subject_ids "${train_ids[@]}" \
  --test_subject_ids "$TARGET" \
  --select_best_on test \
  --softplus \
  --num_epochs 50 \
  --image_feature_dir "$IMAGE_FEATURE_DIR" \
  --text_feature_dir "" \
  --eeg_data_dir "$EEG_DATA_DIR" \
  --device "$DEVICE" \
  --output_dir "$fold_root" \
  --img_l2norm \
  --projector linear \
  --feature_dim 512 \
  --eeg_backbone_dim 1024 \
  --data_average \
  --save_weights \
  --seed 3300 \
  --multi_positive_loss \
  --grouped_batch_sampler \
  --samples_per_image 9 \
  --subject_mixup_mode raw_eeg \
  --mixup_type pairwise \
  --subject_mixup_alpha 0.5

mapfile -t run_dirs < <(
  find "$fold_root" -mindepth 1 -maxdepth 1 -type d -name '*-model' \
    -exec test -f '{}/checkpoint_test_best.pth' ';' -print | sort
)
if [ "${#run_dirs[@]}" -ne 1 ]; then
  echo "expected exactly one completed model directory, found ${#run_dirs[@]}" >&2
  exit 2
fi

dump="$fold_root/embeddings.npz"
if [ ! -f "$dump" ]; then
  python evaluate.py \
    --checkpoint_dir "${run_dirs[0]}" \
    --output_dir "$fold_root/evaluation" \
    --output_name plain_export \
    --eval_mode plain_cosine \
    --test_subject_id "$TARGET" \
    --device "$DEVICE" \
    --batch_size 200 \
    --num_workers "$NW" \
    --dump_npz "$dump"
fi

flock "$RESULT_ROOT/.aggregate.lock" \
  python -m ensemble_experiments.analyze_subject_cohort_bagging \
    --group-mode "$GROUP_MODE" \
    --result-root "$RESULT_ROOT"
