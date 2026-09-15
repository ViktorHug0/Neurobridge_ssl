#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"

DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
# EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-128}"
SEEDS=(${SEEDS:-3300})
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
ALL_SUBJECTS="${ALL_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-0.05}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.5}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-holdout_probe_projector_mixup_$(date +'%Y%m%d-%H%M%S')}"

read -r -a SUBJECT_ARR <<< "${SUBJECTS}"
read -r -a ALL_SUBJECT_ARR <<< "${ALL_SUBJECTS}"

SESSION_DIR="${OUTPUT_DIR_BASE}/${RUN_TAG}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "${SESSION_DIR}"

BASE_EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --subject_probe_holdout --subject_probe_holdout_ratio ${HOLDOUT_RATIO} --subject_probe_interval 5 --subject_probe_layers temporal spatial backbone align --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50"

CONFIG_NAMES=(
    "nomix_featdim512"
    "mixup_featdim512"
)

CONFIG_ARGS=(
    "--subject_mixup_mode none --feature_dim 512 --eeg_backbone_dim 1024"
    "--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha ${MIXUP_ALPHA} --feature_dim 512 --eeg_backbone_dim 1024"
)

append_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$SESSION_SUMMARY" "$config_name" "$run_dir" <<'PY'
import os
import pandas as pd
import sys

summary_file, config_name, run_dir = sys.argv[1:4]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")

if not os.path.exists(avg_csv):
    raise SystemExit(0)

df = pd.read_csv(avg_csv)
avg_row = df[df["sub"] == "Average"].copy()
if avg_row.empty:
    raise SystemExit(0)

avg_row.insert(0, "config", config_name)
if os.path.exists(summary_file):
    prev_df = pd.read_csv(summary_file)
    prev_df = prev_df[prev_df["config"] != config_name]
    avg_row = pd.concat([prev_df, avg_row], ignore_index=True)
avg_row.to_csv(summary_file, index=False)
PY
}

aggregate_probe_metrics() {
    local run_dir="$1"
    python3 - "$run_dir" <<'PY'
import os
import pandas as pd
import sys

run_dir = sys.argv[1]
probe_dfs = []

for subdir in sorted(os.listdir(run_dir)):
    exp_dir = os.path.join(run_dir, subdir)
    probe_csv = os.path.join(exp_dir, "probe_metrics.csv")
    if not os.path.isfile(probe_csv):
        continue
    df = pd.read_csv(probe_csv)
    df.insert(0, "sub", subdir[-6:])
    probe_dfs.append(df)

if not probe_dfs:
    raise SystemExit(0)

all_df = pd.concat(probe_dfs, ignore_index=True)
all_df.to_csv(os.path.join(run_dir, "probe_metrics_all_subjects.csv"), index=False)

epoch_avg = (
    all_df.groupby("epoch", as_index=False)[["eeg_backbone_val_acc", "eeg_align_val_acc"]]
    .mean()
)
epoch_avg.to_csv(os.path.join(run_dir, "probe_metrics_epoch_average.csv"), index=False)
PY
}

for SEED in "${SEEDS[@]}"
do
    for c_idx in "${!CONFIG_NAMES[@]}"
    do
        CONFIG_NAME="${CONFIG_NAMES[$c_idx]}_seed${SEED}"
        EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
        RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
        mkdir -p "${RUN_DIR}"

        echo "=========================================================="
        echo "Running config: ${CONFIG_NAME}"
        echo "Args: ${EXTRA_ARGS}"
        echo "Output: ${RUN_DIR}"
        echo "=========================================================="

        for SUB_ID in "${SUBJECT_ARR[@]}"
        do
            OUTPUT_NAME=$(printf "sub-%02d" "${SUB_ID}")
            if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
                echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
                continue
            fi

            TRAIN_IDS=()
            for TRAIN_SUB_ID in "${ALL_SUBJECT_ARR[@]}"
            do
                if [ "${TRAIN_SUB_ID}" -ne "${SUB_ID}" ]; then
                    TRAIN_IDS+=("${TRAIN_SUB_ID}")
                fi
            done

            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python3 "${REPO_ROOT}/train.py" \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --learning_rate "${LEARNING_RATE}" \
                --output_name "${OUTPUT_NAME}" \
                --eeg_encoder_type "${EEG_ENCODER_TYPE}" \
                --train_subject_ids "${TRAIN_IDS[@]}" \
                --test_subject_ids "${SUB_ID}" \
                --select_best_on test \
                --softplus \
                --num_epochs "${NUM_EPOCHS}" \
                --image_feature_dir "${IMAGE_FEATURE_DIR}" \
                --text_feature_dir "${TEXT_FEATURE_DIR}" \
                --eeg_data_dir "${EEG_DATA_DIR}" \
                --device "${DEVICE}" \
                --output_dir "${RUN_DIR}" \
                --img_l2norm \
                --projector "${PROJECTOR}" \
                --data_average \
                --save_weights \
                --seed "${SEED}" \
                ${BASE_EXTRA_ARGS} \
                ${EXTRA_ARGS}
        done

        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "${RUN_DIR}" --output_name "inter_subject_summary.csv"
        aggregate_probe_metrics "${RUN_DIR}"
        append_average_row "${CONFIG_NAME}" "${RUN_DIR}"
    done
done

echo "Holdout probe projector/mixup sweep completed: ${SESSION_SUMMARY}"
