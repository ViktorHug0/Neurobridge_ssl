#!/bin/bash
# Train/export one member of the fixed balanced 7-of-9 subject-bag committee.
set -euo pipefail

: "${TARGET:?TARGET is required}"
: "${MEMBER:?MEMBER is required}"

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/subject_cohort_bagging/testselected_balanced7x8}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-6}
MODEL_SEED=${MODEL_SEED:-$((3300 + MEMBER))}

target_tag=$(printf '%02d' "$TARGET")
fold_root="$RESULT_ROOT/target-$target_tag/member-$MEMBER"
mkdir -p "$fold_root"
read -ra train_ids <<< "$(
  python -m ensemble_experiments.balanced_subject_bagging \
    --print-bag "$TARGET" "$MEMBER"
)"

find_completed_runs() {
  mapfile -t candidate_run_dirs < <(
    find "$fold_root" -mindepth 1 -maxdepth 1 -type d -name '*-model' | sort
  )
  run_dirs=()
  for candidate_run_dir in "${candidate_run_dirs[@]}"; do
    if [ -f "$candidate_run_dir/checkpoint_test_best.pth" ] && \
       [ -f "$candidate_run_dir/result.csv" ]; then
      run_dirs+=("$candidate_run_dir")
    fi
  done
}

find_completed_runs
if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "[balanced-bag] target=$TARGET member=$MEMBER seed=$MODEL_SEED train=${train_ids[*]}"
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
    --seed "$MODEL_SEED" \
    --multi_positive_loss \
    --grouped_batch_sampler \
    --samples_per_image 7 \
    --subject_mixup_mode raw_eeg \
    --mixup_type pairwise \
    --subject_mixup_alpha 0.5
  find_completed_runs
fi
if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "no completed model directory found under $fold_root" >&2
  exit 2
fi
if [ "${#run_dirs[@]}" -gt 1 ]; then
  echo "warning: found ${#run_dirs[@]} completed retries; using earliest: ${run_dirs[0]}" >&2
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
  python -m ensemble_experiments.analyze_balanced_subject_bagging \
    --result-root "$RESULT_ROOT"
