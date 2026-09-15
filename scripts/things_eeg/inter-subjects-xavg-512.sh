#!/bin/bash
# Cross-subject trial-averaging augmentation, featdim-512, L28 target.
# One synthetic trial per image = mean of a random-size (Beta-drawn) random subset of the
# raw reps pooled across all training subjects. Standard InfoNCE (no multi-positive/mixup).
#
# Epoch here is ~16 batches (16540 images / 1024) vs ~145 for the multi-positive baseline,
# so EPOCHS defaults to 450 to match total gradient steps (~50 baseline epochs).
# Arms: uniform (1,1), low-skew (1,3), high-skew (3,1). Holds out subjects 1..7.
set -e
trap 'echo "Script Error"' ERR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"

IMAGE_FEATURE_DIR="${REPO_ROOT}/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"; SEED="${SEED:-3300}"; EPOCHS="${EPOCHS:-450}"
HELDOUT="${HELDOUT:-1 2 3 4 5 6 7}"
# ARMS: space-separated "a,b" Beta pairs. Default = uniform, low-skew, high-skew.
ARMS="${ARMS:-1,1 1,3 3,1}"

for ARM in $ARMS; do
    A="${ARM%,*}"; B="${ARM#*,}"
    SESSION_DIR="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}/xavg512_beta${A}-${B}_$(date +%Y%m%d-%H%M%S)/seed${SEED}"
    mkdir -p "$SESSION_DIR"
    for SUB_ID in $HELDOUT; do
        TRAIN_IDS=""; for i in {1..10}; do [ "$i" -ne "$SUB_ID" ] && TRAIN_IDS+="$i "; done
        echo "### xavg beta=(${A},${B}) held-out subject ${SUB_ID} ###"
        python3 train.py \
            --batch_size 1024 --num_workers 4 --learning_rate 3e-4 --num_epochs "$EPOCHS" \
            --output_name "$(printf 'sub-%02d' "$SUB_ID")" --output_dir "$SESSION_DIR" \
            --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
            --train_subject_ids $TRAIN_IDS --test_subject_ids "$SUB_ID" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" \
            --cross_subject_average --xavg_beta_a "$A" --xavg_beta_b "$B" \
            --xavg_kmin 4 --xavg_kmax 36 \
            --eeg_data_dir "$EEG_DATA_DIR" --device "$DEVICE" \
            --feature_dim 512 --eeg_backbone_dim 1024 \
            --softplus --img_l2norm --projector linear --save_weights \
            --text_feature_dir "" --seed "$SEED"
        python3 compute_avg_results.py --result_dir "$SESSION_DIR" --output_name "inter_subject_summary.csv"
    done
    echo "DONE arm (${A},${B}): $SESSION_DIR"
done
