#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi
cd "$REPO_ROOT"

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-512}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-1024}"
SEEDS=(3300 3301 3302)
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
ALL_SUBJECTS="${ALL_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SELECTED_CHANNELS=()

read -r -a SUBJECT_ARR <<< "$SUBJECTS"
read -r -a ALL_SUBJECT_ARR <<< "$ALL_SUBJECTS"

CONFIG_NAMES=(
    "param_k30_pool51_do050"
)

CONFIG_ARGS=(
    "--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50"
)

SESSION_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
SESSION_DIR="${OUTPUT_DIR_BASE}/tsconv_dropout_sweep_${SESSION_TIMESTAMP}" # 20260429-170207" # 20260428-142358" # 20260426-133443"
SESSION_SUMMARY="${SESSION_DIR}/dropout_sweep_summary.csv"
mkdir -p "$SESSION_DIR"

upsert_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$run_dir" "$config_name" "$SESSION_SUMMARY" <<'PY'
import os
import pandas as pd
import sys

run_dir, config_name, session_summary_path = sys.argv[1:4]
inter_summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
if not os.path.exists(inter_summary_path):
    raise SystemExit(0)

df = pd.read_csv(inter_summary_path)
avg_row = df[df["sub"] == "Average"].copy()
if avg_row.empty:
    raise SystemExit(0)

avg_row.insert(0, "config", config_name)
if os.path.exists(session_summary_path):
    session_df = pd.read_csv(session_summary_path)
    session_df = session_df[session_df["config"] != config_name]
    avg_row = pd.concat([session_df, avg_row], ignore_index=True)
avg_row.to_csv(session_summary_path, index=False)
PY
}

for SEED in "${SEEDS[@]}"
do
    for c_idx in "${!CONFIG_NAMES[@]}"
    do
        CONFIG_NAME="${CONFIG_NAMES[$c_idx]}_featdim${FEATURE_DIM}_seed${SEED}"
        EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
        RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
        mkdir -p "$RUN_DIR"

        echo "=========================================================="
        echo "Running config: ${CONFIG_NAME} (Seed: ${SEED})"
        echo "Args: ${EXTRA_ARGS}"
        echo "Output: ${RUN_DIR}"
        echo "=========================================================="

        for SUB_ID in "${SUBJECT_ARR[@]}"
        do
            OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
            if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
                echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
                continue
            fi

            TRAIN_IDS=()
            for TRAIN_SUB_ID in "${ALL_SUBJECT_ARR[@]}"
            do
                if [ "$TRAIN_SUB_ID" -ne "$SUB_ID" ]; then
                    TRAIN_IDS+=("$TRAIN_SUB_ID")
                fi
            done

            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python3 train.py \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --learning_rate "$LEARNING_RATE" \
                --output_name "$OUTPUT_NAME" \
                --train_subject_ids "${TRAIN_IDS[@]}" \
                --test_subject_ids "$SUB_ID" \
                --select_best_on test \
                --softplus \
                --num_epochs "$NUM_EPOCHS" \
                --image_feature_dir "$IMAGE_FEATURE_DIR" \
                --text_feature_dir "$TEXT_FEATURE_DIR" \
                --eeg_data_dir "$EEG_DATA_DIR" \
                --device "$DEVICE" \
                --output_dir "$RUN_DIR" \
                --selected_channels "${SELECTED_CHANNELS[@]}" \
                --img_l2norm \
                --projector "$PROJECTOR" \
                --feature_dim "$FEATURE_DIM" \
                --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
                --data_average \
                --save_weights \
                --multi_positive_loss \
                --grouped_batch_sampler \
                --samples_per_image 9 \
                --subject_mixup_mode raw_eeg \
                --mixup_type pairwise \
                --subject_mixup_alpha 0.5 \
                --seed "$SEED" \
                $EXTRA_ARGS
        done

        python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
        upsert_average_row "$CONFIG_NAME" "$RUN_DIR"
    done
done

echo "TSConv encoder sweep completed: ${SESSION_SUMMARY}"
