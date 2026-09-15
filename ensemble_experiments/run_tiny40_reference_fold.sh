#!/bin/bash
set -euo pipefail

: "${TARGET:?TARGET is required}"
: "${MEMBER:?MEMBER is required (0=TinyTSConv, 1=TinyATM)}"

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/tiny_reference_pair_40e/testselected_internvit28}
EEG_DATA_DIR=${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}
IMAGE_FEATURE_DIR=${IMAGE_FEATURE_DIR:-$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}
DEVICE=${DEVICE:-cuda:0}
NW=${NW:-2}
AMP_DTYPE=${AMP_DTYPE:-none}
RUN_TAG=${RUN_TAG:-tiny40}

cd "$REPO_ROOT"
source .venv/bin/activate

case "$MEMBER" in
  0)
    member_name=${RUN_TAG}_tsconv_seed3300
    model_seed=3300
    model_args=(
      --eeg_encoder_type TSConv_parameterizable
      --eeg_backbone_dim 256
      --tsconv_temporal_filters 10
      --tsconv_temporal_kernel 25
      --tsconv_pool_kernel 51
      --tsconv_pool_stride 5
      --tsconv_spatial_filters 10
      --tsconv_projection_filters 10
    )
    ;;
  1)
    member_name=${RUN_TAG}_atm_seed4300
    model_seed=4300
    model_args=(
      --eeg_encoder_type ATM
      --eeg_backbone_dim 128
      --atm_d_model 112
      --atm_n_heads 4
      --atm_e_layers 1
      --atm_d_ff 112
      --atm_temporal_filters 12
      --atm_temporal_kernel 25
      --atm_pool_kernel 51
      --atm_pool_stride 5
      --atm_spatial_filters 12
      --atm_projection_filters 12
    )
    ;;
  *)
    echo "invalid MEMBER=$MEMBER" >&2
    exit 2
    ;;
esac

target_tag=$(printf '%02d' "$TARGET")
fold_root="$RESULT_ROOT/$member_name/sub-$target_tag"
dump="$fold_root/embeddings.npz"
mkdir -p "$fold_root"

if [ -f "$dump" ]; then
  echo "[tiny40] existing export found; skipping member=$MEMBER target=$TARGET"
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
    --num_epochs 40 \
    --image_feature_dir "$IMAGE_FEATURE_DIR" \
    --text_feature_dir "" \
    --eeg_data_dir "$EEG_DATA_DIR" \
    --device "$DEVICE" \
    --amp_dtype "$AMP_DTYPE" \
    --output_dir "$fold_root" \
    --img_l2norm \
    --projector linear \
    --feature_dim 128 \
    --data_average \
    --save_weights \
    --seed "$model_seed" \
    --multi_positive_loss \
    --grouped_batch_sampler \
    --samples_per_image 9 \
    --subject_mixup_mode raw_eeg \
    --mixup_type pairwise \
    --subject_mixup_alpha 0.5 \
    "${model_args[@]}"
fi

mapfile -t run_dirs < <(find_completed_runs)
if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "no completed Tiny-40 run found" >&2
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
