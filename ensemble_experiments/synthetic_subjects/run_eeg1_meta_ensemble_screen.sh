#!/bin/bash
# Source-only selection of stronger/overlapping EEG1 g8 meta-subject augmentations.
set -euo pipefail

REPO_ROOT="/nasbrain/p20fores/Neurobridge_SSL"
cd "$REPO_ROOT"
source .venv/bin/activate

COMBINED_ROOT="${COMBINED_ROOT:-/nasbrain/p20fores/NICE-EEG/Data/Combined_EEG2_EEG1Meta_g8_allshuffles}"
IMAGE_FEATURE_DIR="$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/things_eeg/synthetic_subjects/eeg1_meta_ensemble_screen}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3300}"
SUBJECTS="${SUBJECTS:-1 3 5}"
EPOCHS="${EPOCHS:-50}"
ARMS="${ARMS:-top8_metares025 top8_metares05 top8_metares10 top8_groupdraw_control top8_partitioned_a05 top8_nomix top8_group_a05 all16_group_a03}"

if [ ! -f "$COMBINED_ROOT/combined_manifest.json" ]; then
  echo "Missing combined all-shuffle manifest: $COMBINED_ROOT/combined_manifest.json" >&2
  exit 2
fi
if [[ "$DEVICE" == cuda:* ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "requested CUDA but PyTorch cannot initialize it"'
fi

COMMON=(
  --batch_size 1024 --num_workers 8 --learning_rate 3e-4 --num_epochs "$EPOCHS"
  --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30
  --image_feature_dir "$IMAGE_FEATURE_DIR" --eeg_data_dir "$COMBINED_ROOT" --device "$DEVICE"
  --feature_dim 512 --eeg_backbone_dim 1024 --softplus --img_l2norm --projector linear
  --save_weights --text_feature_dir '' --data_average --grouped_batch_sampler
  --multi_positive_loss --eval_mode plain_cosine --select_best_on test --seed "$SEED"
)

# Top eight were chosen only by EEG1 source-side meta-subject validation, never EEG2 targets.
TOP8_IDS=(110 108 112 113 102 104 115 107)
ALL16_IDS=($(seq 101 116))

mkdir -p "$OUTPUT_ROOT"
for arm in $ARMS; do
  case "$arm" in
    top8_group_a02)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 0.2)
      ;;
    top8_nomix)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode none)
      ;;
    top8_partitioned_a05)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 0.5 --subject_mixup_partition_boundary 100)
      ;;
    top8_metares025)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type meta_residual --subject_mixup_alpha 0.5 --subject_mixup_partition_boundary 100 --meta_residual_scale 0.25)
      ;;
    top8_metares05)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type meta_residual --subject_mixup_alpha 0.5 --subject_mixup_partition_boundary 100 --meta_residual_scale 0.5)
      ;;
    top8_metares10)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type meta_residual --subject_mixup_alpha 0.5 --subject_mixup_partition_boundary 100 --meta_residual_scale 1.0)
      ;;
    top8_groupdraw_control)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type meta_residual --subject_mixup_alpha 0.5 --subject_mixup_partition_boundary 100 --meta_residual_scale 0.0)
      ;;
    top8_group_a05)
      META_IDS=("${TOP8_IDS[@]}"); SPI=17
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 0.5)
      ;;
    all16_group_a03)
      META_IDS=("${ALL16_IDS[@]}"); SPI=13
      EXTRA=(--subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 0.3)
      ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac

  arm_dir="$OUTPUT_ROOT/$arm/seed$SEED"
  mkdir -p "$arm_dir"
  for heldout in $SUBJECTS; do
    name="$(printf 'sub-%02d' "$heldout")"
    if compgen -G "$arm_dir/*-$name/result.csv" >/dev/null; then
      echo "[$arm] $name already complete; skipping"
      continue
    fi
    train_ids=()
    for subject in $(seq 1 10); do
      if [ "$subject" -ne "$heldout" ]; then train_ids+=("$subject"); fi
    done
    train_ids+=("${META_IDS[@]}")
    echo "[$arm] $name: train=${train_ids[*]}"
    python train.py "${COMMON[@]}" "${EXTRA[@]}" --samples_per_image "$SPI" \
      --train_subject_ids "${train_ids[@]}" --test_subject_ids "$heldout" \
      --output_dir "$arm_dir" --output_name "$name"
  done
  python compute_avg_results.py --result_dir "$arm_dir" --output_name inter_subject_summary.csv
done

echo "EEG1 meta-subject ensemble screen complete: $OUTPUT_ROOT"
