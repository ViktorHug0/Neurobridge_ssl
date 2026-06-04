#!/bin/bash
# Sparse CLIP inter-subject LOSO: bb64 + feature_dim 512 + ReLU alignment projectors.
# Baseline for comparison: projector_only_bb64_sweep featdim_512_seed3300 (dense, no ReLU).
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature}"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"

DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-2048}"
EEG_BACKBONE_DIM_FIXED="${EEG_BACKBONE_DIM_FIXED:-256}"
SEED="${SEED:-3300}"

# Optional smoke: SMOKE_SUBJECTS="6 10" runs only those held-out subjects.
SMOKE_SUBJECTS="${SMOKE_SUBJECTS:-1}"

OUTPUT_DIR_BASE="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subject-sparse}"
SESSION_TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
SESSION_DIR="${SESSION_DIR:-${OUTPUT_DIR_BASE}/sparse_clip_bb${EEG_BACKBONE_DIM_FIXED}_fd${FEATURE_DIM}_seed${SEED}_${SESSION_TIMESTAMP}}"
CONFIG_NAME="sparse_relu_fd${FEATURE_DIM}_seed${SEED}"
RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "$RUN_DIR"

# BASE_EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --eeg_backbone_dim ${EEG_BACKBONE_DIM_FIXED} --projector_activation relu"
BASE_EXTRA_ARGS="--eeg_backbone_dim ${EEG_BACKBONE_DIM_FIXED} --projector_activation relu"

echo "=============================================================="
echo "Sparse CLIP inter-subject | bb=${EEG_BACKBONE_DIM_FIXED} fd=${FEATURE_DIM} relu"
echo "SESSION_DIR: $SESSION_DIR"
echo "SEED: $SEED | NUM_EPOCHS: $NUM_EPOCHS"
if [ -n "$SMOKE_SUBJECTS" ]; then
    echo "SMOKE_SUBJECTS: $SMOKE_SUBJECTS"
fi
echo "=============================================================="

if [ -n "$SMOKE_SUBJECTS" ]; then
    SUBJECT_LOOP=($SMOKE_SUBJECTS)
else
    SUBJECT_LOOP=({1..10})
fi

for SUB_ID in "${SUBJECT_LOOP[@]}"
do
    OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
    if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
        echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
        continue
    fi
    echo "Training subject ${SUB_ID} for $CONFIG_NAME..."

    TRAIN_IDS=""
    for i in {1..10}
    do
        if [ "$i" -ne "$SUB_ID" ]; then
            TRAIN_IDS+="$i "
        fi
    done

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 train.py \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $TRAIN_IDS \
        --test_subject_ids "$SUB_ID" \
        --select_best_on test \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE" \
        --output_dir "$RUN_DIR" \
        --img_l2norm \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --data_average \
        --save_weights \
        --seed "$SEED" \
        $BASE_EXTRA_ARGS
done

python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"

echo "Collecting average results for $CONFIG_NAME into $SESSION_SUMMARY..."
python3 - "$SESSION_SUMMARY" "$CONFIG_NAME" "$RUN_DIR" <<'PY'
import pandas as pd
import os
import sys

summary_file, config_name, run_dir = sys.argv[1:4]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")

if os.path.exists(avg_csv):
    df = pd.read_csv(avg_csv)
    avg_row = df[df["sub"] == "Average"].copy()
    if not avg_row.empty:
        avg_row.insert(0, "config", config_name)
        write_header = not os.path.exists(summary_file)
        avg_row.to_csv(summary_file, mode="a", header=write_header, index=False)
PY

echo "Sparse CLIP sweep completed: $SESSION_SUMMARY"
