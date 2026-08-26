#!/bin/bash
# Test-selected nine-source LOSO run matching the group_e100 recipe behind the
# 45.35% ensemble.  FOLDS may be used for a cheap promotion gate.
set -euo pipefail

ARM=${1:?usage: run_group_target_testselected.sh <arm> <image-feature-dir>}
FEATURES=${2:?usage: run_group_target_testselected.sh <arm> <image-feature-dir>}

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

SEED=${RUN_SEED:-3300}
NUM_EPOCHS=${NUM_EPOCHS:-100}
OUT=${OUT_OVERRIDE:-"$REPO_ROOT/results/things_eeg/ensemble50_testselected/$ARM/seed$SEED"}
FOLDS=${FOLDS:-"1 2 3 4 5 6 7 8 9 10"}
ENCODER=${ENCODER:-TSConv_parameterizable}
FEATURE_DIM=${FEATURE_DIM:-512}
BACKBONE_DIM=${BACKBONE_DIM:-1024}
MIXUP_TYPE=${MIXUP_TYPE:-group}
TKERNEL=${TKERNEL:-30}
mkdir -p "$OUT"

for held in $FOLDS; do
  tag=$(printf '%02d' "$held")
  train_ids=()
  for subject in $(seq 1 10); do
    [ "$subject" -ne "$held" ] && train_ids+=("$subject")
  done
  python train.py \
    --batch_size 1024 --num_workers "${NW:-6}" --learning_rate 3e-4 --num_epochs "$NUM_EPOCHS" \
    --output_name "sub-$tag" --output_dir "$OUT" \
    --eeg_encoder_type "$ENCODER" --tsconv_temporal_kernel "$TKERNEL" \
    --train_subject_ids "${train_ids[@]}" --test_subject_ids "$held" \
    --image_feature_dir "$FEATURES" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
    --device cuda:0 --feature_dim "$FEATURE_DIM" --eeg_backbone_dim "$BACKBONE_DIM" \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --samples_per_image 9 --multi_positive_loss \
    --subject_mixup_mode raw_eeg --mixup_type "$MIXUP_TYPE" --subject_mixup_alpha 0.5 \
    --eval_mode plain_cosine --select_best_on test --seed "$SEED"
done

python compute_avg_results.py \
  --result_dir "$OUT" --output_name inter_subject_summary.csv
