#!/bin/bash
set -euo pipefail

: "${TARGET:?TARGET is required}"

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/tiny_multiscale_mixer/testselected_internvit28}
readonly EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
readonly IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
readonly DEVICE=${DEVICE:-cuda:0}
readonly NW=${NW:-1}
readonly MODEL_SEED=3300
readonly MEMBER_NAME=tiny_multiscale_mixer_seed3300

cd "$REPO_ROOT"
source .venv/bin/activate

target_tag=$(printf '%02d' "$TARGET")
fold_root="$RESULT_ROOT/$MEMBER_NAME/sub-$target_tag"
dump="$fold_root/embeddings.npz"
mkdir -p "$fold_root"

if [ -f "$dump" ]; then
  echo "[tiny-mixer] existing export found; skipping target=$TARGET"
  exit 0
fi

find_completed_runs() {
  find "$fold_root" -mindepth 1 -maxdepth 1 -type d -name '*-model' \
    -exec test -f '{}/checkpoint_test_best.pth' ';' \
    -exec test -f '{}/result.csv' ';' -print | sort
}

mapfile -t run_dirs < <(find_completed_runs)
train_ids=()
for subject in $(seq 1 10); do
  if [ "$subject" -ne "$TARGET" ]; then
    train_ids+=("$subject")
  fi
done

if [ "${#run_dirs[@]}" -eq 0 ]; then
  python train.py \
    --batch_size 1024 \
    --num_workers "$NW" \
    --learning_rate 3e-4 \
    --output_name model \
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
    --feature_dim 128 \
    --eeg_backbone_dim 128 \
    --data_average \
    --save_weights \
    --seed "$MODEL_SEED" \
    --multi_positive_loss \
    --grouped_batch_sampler \
    --samples_per_image 9 \
    --subject_mixup_mode raw_eeg \
    --mixup_type pairwise \
    --subject_mixup_alpha 0.5 \
    --eeg_encoder_type TinyMultiScaleTSMixer
fi

mapfile -t run_dirs < <(find_completed_runs)
if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "no completed TinyMultiScaleTSMixer run found" >&2
  exit 2
fi
run_dir=${run_dirs[$((${#run_dirs[@]} - 1))]}

python evaluate.py \
  --checkpoint_dir "$run_dir" \
  --output_dir "$fold_root/evaluation" \
  --output_name plain_export \
  --eval_mode plain_cosine \
  --test_subject_id "$TARGET" \
  --device "$DEVICE" \
  --batch_size 200 \
  --num_workers "$NW" \
  --dump_npz "$dump"

