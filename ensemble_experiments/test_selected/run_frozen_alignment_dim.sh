#!/bin/bash
# Controlled alignment-dimension ensemble: every branch starts from the same
# fold-specific TSConv seed-3300 encoder, replaces both projectors, freezes the
# encoder in eval mode, and trains the fresh projectors for exactly ten epochs.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

REFERENCE_ROOT="$REPO_ROOT/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300"
OUTPUT_ROOT="$REPO_ROOT/results/things_eeg/ensemble_alignment_dim_frozen/tsconv_pair_seed3300"
IMAGE_FEATURE_DIR="$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
DIMS=${DIMS:-"32 64 128 256 512 1024"}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10"}

for dim in $DIMS; do
  run_root="$OUTPUT_ROOT/fd$dim"
  mkdir -p "$run_root"

  for held in $SUBJECTS; do
    tag=$(printf '%02d' "$held")
    if compgen -G "$run_root/*-sub-$tag/result.csv" > /dev/null; then
      echo "[skip] fd=$dim fold=$tag already complete"
      continue
    fi

    mapfile -t checkpoints < <(
      find "$REFERENCE_ROOT" -mindepth 2 -maxdepth 2 -type f \
        -path "*-sub-$tag/checkpoint_test_best.pth" | sort
    )
    if [ "${#checkpoints[@]}" -ne 1 ]; then
      echo "Expected exactly one reference checkpoint for sub-$tag; got ${#checkpoints[@]}" >&2
      printf '%s\n' "${checkpoints[@]}" >&2
      exit 1
    fi

    train_ids=()
    for subject in $(seq 1 10); do
      [ "$subject" -ne "$held" ] && train_ids+=("$subject")
    done

    python train.py \
      --batch_size 1024 --num_workers 6 --learning_rate 3e-4 --weight_decay 1e-4 \
      --num_epochs 10 --output_name "sub-$tag" --output_dir "$run_root" \
      --eeg_encoder_type TSConv_parameterizable \
      --tsconv_temporal_filters 40 --tsconv_temporal_kernel 30 \
      --tsconv_pool_kernel 51 --tsconv_pool_stride 5 \
      --tsconv_spatial_filters 40 --tsconv_projection_filters 40 \
      --tsconv_activation elu --tsconv_dropout 0.5 \
      --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
      --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$EEG_DATA_DIR" \
      --device cuda:0 --feature_dim "$dim" --eeg_backbone_dim 1024 \
      --projector linear --softplus --img_l2norm --text_feature_dir '' \
      --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
      --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5 \
      --init_checkpoint "${checkpoints[0]}" --init_checkpoint_scope encoder \
      --freeze_eeg_encoder --train_rng_seed 7330 \
      --eval_mode plain_cosine --select_best_on test --save_weights --seed 3300
  done

  python compute_avg_results.py \
    --result_dir "$run_root" --output_name inter_subject_summary.csv
  python ensemble_experiments/test_selected/dump_testselected_arm.py \
    --arm "dimfrozen_fd$dim" --run-root "$run_root" \
    --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
done

python ensemble_experiments/analysis/analyze_alignment_dim_ensemble.py \
  --output ensemble_experiments/analysis/frozen_alignment_dim_ensemble.json
