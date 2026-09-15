#!/bin/bash
# EEG foundation-model LOSO driver (reviewer vxam-Q1): full-parameter fine-tuning of a pretrained
# LaBraM or CBraMod in place of TSConv, under the AVDE contrastive-plus-regression objective.
#
# Reconstructed 2026-09-15; the original was lost while scripts/ was gitignored (see PROTECTED.md).
# Every flag below is taken from the surviving train_config.json of
# results/things_eeg/foundation/{LaBraM,CBraMod}_{mix,nomix}_seed3300/. The encoder wrapper this
# calls (module/eeg_encoder/foundation.py) reproduces all four recorded arms exactly on sub-01:
#   LaBraM_mix 30.00/59.50   LaBraM_nomix 25.50/57.00
#   CBraMod_mix 11.50/30.50  CBraMod_nomix 8.50/28.00
#
#   ARM=LaBraM_mix bash scripts/things_eeg/inter-subject-foundation.sh
#   ARM=CBraMod_nomix SUBJECTS="1 2 3" bash scripts/things_eeg/inter-subject-foundation.sh
#
# Cost was ~1.4 h/fold at 50 epochs on the shared GPU, so run it under sbatch and respect the
# two-concurrent-GPU-job cap.
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${REPO_ROOT}/.venv/bin/python"

# braindecode fetches the pretrained checkpoints from this cache; both are already present.
export HF_HOME="${HF_HOME:-/nasbrain/p20fores/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ARM="${ARM:-LaBraM_mix}"
case "$ARM" in
    LaBraM_mix)     ENCODER=LaBraM;  MIX=1 ;;
    LaBraM_nomix)   ENCODER=LaBraM;  MIX=0 ;;
    CBraMod_mix)    ENCODER=CBraMod; MIX=1 ;;
    CBraMod_nomix)  ENCODER=CBraMod; MIX=0 ;;
    *) echo "ARM must be one of LaBraM_mix LaBraM_nomix CBraMod_mix CBraMod_nomix"; exit 1 ;;
esac

IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-${REPO_ROOT}/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3300}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/foundation}"
RUN_DIR="${OUTPUT_DIR_BASE}/${ARM}_seed${SEED}"
mkdir -p "$RUN_DIR"

# AVDE full-FT recipe, verbatim from the recorded train_config.json.
MIX_ARGS=""
if [ "$MIX" = "1" ]; then
    MIX_ARGS="--subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 --mixup_type pairwise"
fi

for SUB_ID in $SUBJECTS; do
    OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
    TRAIN_IDS=""
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if [ "$i" -ne "$SUB_ID" ]; then TRAIN_IDS+="$i "; fi
    done

    echo "=== ${ARM}: training subject ${SUB_ID} ==="
    "$PYTHON" train.py \
        --eeg_encoder_type "$ENCODER" \
        --train_subject_ids $TRAIN_IDS \
        --test_subject_ids "$SUB_ID" \
        --output_name "$OUTPUT_NAME" \
        --output_dir "$RUN_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --eeg_backbone_dim 1024 \
        --feature_dim 512 \
        --projector linear \
        --multi_positive_loss \
        --grouped_batch_sampler \
        --samples_per_image 9 \
        --data_average \
        --img_l2norm \
        --softplus \
        --alpha 0.8 \
        --batch_size 256 \
        --learning_rate 2e-3 \
        --lr_scheduler cosine \
        --warmup_epochs 5 \
        --min_lr 1e-5 \
        --weight_decay 0.05 \
        --num_epochs 50 \
        --time_window 0 250 \
        --device "$DEVICE" \
        --save_weights \
        $MIX_ARGS \
        --seed "$SEED"

    "$PYTHON" compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
done

echo "done: $RUN_DIR"
