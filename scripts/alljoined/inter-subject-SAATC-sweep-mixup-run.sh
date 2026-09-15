#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Workaround for "Key has expired" (ENOKEY) errors on home directory
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-${REPO_ROOT}/data/things_eeg/image_feature}"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_DIR:-${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}}"
EEG_DATA_DIR="${EEG_DATA_DIR:-${REPO_ROOT}/data/alljoined_final/preprocessed_eeg}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/alljoined/inter-subjects}"
RUN_TAG="${RUN_TAG:-sattc_final_paper_512_sweep_run_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_sweep_summary.csv"
# Root folder containing one or more trained source configs.
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/results/alljoined/inter-subjects}"
SOURCE_CONFIG_PATTERNS="${SOURCE_CONFIG_PATTERNS:-tsconv_fd512_mixup_raw_pairwise_linear_a0p5_seed3300 tsconv_fd512_mixup_raw_pairwise_linear_a0p5_seed3301 tsconv_fd512_mixup_raw_pairwise_linear_a0p5_seed3302}"
# saw0p94_k3_tau0p1_steps16_pow1p2_iters12
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
SOFT_STEP_VALUES="${SOFT_STEP_VALUES:-1}" 
SOFT_POWER_VALUES="${SOFT_POWER_VALUES:-1.6}"
SINKHORN_ITER_VALUES="${SINKHORN_ITER_VALUES:-5}"
TAU_VALUES="${TAU_VALUES:-0.08}"
K_VALUES="${K_VALUES:-3}"
SAW_SHRINK_VALUES="${SAW_SHRINK_VALUES:-0.98}"
ALIGN_SUBSPACE_DIM="${ALIGN_SUBSPACE_DIM:-}"

mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

if [ ! -d "$SOURCE_RUN_ROOT" ]; then
    echo "Source run root does not exist: $SOURCE_RUN_ROOT"
    exit 1
fi

append_average_row() {
    local source_label="$1"
    local config_name="$2"
    local run_dir="$3"
    python3 - "$UNIFIED_CSV" "$source_label" "$config_name" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, source_label, config_name, run_dir = sys.argv[1:5]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "source_run": source_label,
    "config": config_name,
    "architecture": "",
    "eval_mode": "",
    "top1 acc": "",
    "top5 acc": "",
    "best top1 acc": "",
    "best top5 acc": "",
    "best test loss": "",
    "best epoch": "",
}

if os.path.isfile(summary_path):
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("sub", "").strip().lower() == "average":
                for key in row:
                    if key in r:
                        row[key] = r[key]
                break

write_header = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

find_checkpoint_dir() {
    local source_run_dir="$1"
    local output_name="$2"
    ls -td "${source_run_dir}"/*-"${output_name}" 2>/dev/null | head -n 1
}

sanitize_tag() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "$value"
}

run_spec_for_source() {
    local source_run_dir="$1"
    local source_label="$2"
    local tag="$3"
    local eval_mode="$4"
    local csls_k="$5"
    local saw_shrink="$6"
    local sinkhorn_tau="$7"
    local sinkhorn_iters="$8"
    local soft_steps="$9"
    local soft_power="${10}"
    local extra_tag=""
    local extra_args=()
    if [ -n "$ALIGN_SUBSPACE_DIM" ]; then
        extra_tag="_subspace$(sanitize_tag "$ALIGN_SUBSPACE_DIM")"
        extra_args+=(--sattc_alignment_subspace_dim "$ALIGN_SUBSPACE_DIM")
    fi

    local config_name="${source_label}_$(sanitize_tag "$tag")${extra_tag}"
    local config_run_dir="${RUN_ROOT}/${source_label}/${config_name}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Sweep: ${config_name}"
    echo "source=${source_run_dir}"
    echo "tag=${tag}${extra_tag} mode=${eval_mode} k=${csls_k} saw=${saw_shrink} sinkhorn=${sinkhorn_tau}/${sinkhorn_iters} soft=${soft_steps}/${soft_power}"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$source_run_dir" "$OUTPUT_NAME")"
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${source_run_dir}"
            exit 1
        fi

        python3 "${REPO_ROOT}/evaluate.py" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$config_run_dir" \
            --output_name "$OUTPUT_NAME" \
            --eval_mode "$eval_mode" \
            --test_subject_id "$SUB_ID" \
            --eeg_data_dir "$EEG_DATA_DIR" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --device "$DEVICE" \
            --sattc_saw_shrink "$saw_shrink" \
            --sattc_csls_k "$csls_k" \
            --sattc_sinkhorn \
            --sattc_sinkhorn_tau "$sinkhorn_tau" \
            --sattc_sinkhorn_iters "$sinkhorn_iters" \
            --sattc_soft_procrustes \
            --sattc_soft_procrustes_steps "$soft_steps" \
            --sattc_soft_procrustes_power "$soft_power" \
            "${extra_args[@]}"
    done

    python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$source_label" "$config_name" "$config_run_dir"
}

shopt -s nullglob
SOURCE_DIRS=()
read -r -a SOURCE_PATTERN_ARR <<< "$SOURCE_CONFIG_PATTERNS"
for PAT in "${SOURCE_PATTERN_ARR[@]}"; do
    for DIR in "${SOURCE_RUN_ROOT}"/${PAT}; do
        if [ -d "$DIR" ]; then
            SOURCE_DIRS+=("$DIR")
        fi
    done
done
shopt -u nullglob

if [ "${#SOURCE_DIRS[@]}" -eq 0 ]; then
    echo "No source config directories matched SOURCE_CONFIG_PATTERNS='${SOURCE_CONFIG_PATTERNS}' under ${SOURCE_RUN_ROOT}"
    exit 1
fi

for SOURCE_RUN_DIR in "${SOURCE_DIRS[@]}"
do
    if [ ! -d "$SOURCE_RUN_DIR" ]; then
        continue
    fi
    SOURCE_LABEL="$(basename "$SOURCE_RUN_DIR")"

    for K in ${K_VALUES} 
    do
        for SAW_SHRINK in ${SAW_SHRINK_VALUES}
        do
            for TAU in ${TAU_VALUES}
            do
                for SOFT_STEPS in ${SOFT_STEP_VALUES}
                do
                    for SOFT_POWER in ${SOFT_POWER_VALUES}
                    do
                        for SINKHORN_ITERS in ${SINKHORN_ITER_VALUES}
                        do
                            TAG="saw${SAW_SHRINK}_k${K}_tau${TAU}_steps${SOFT_STEPS}_pow${SOFT_POWER}_iters${SINKHORN_ITERS}"
                            run_spec_for_source \
                                "$SOURCE_RUN_DIR" \
                                "$SOURCE_LABEL" \
                                "$TAG" \
                                "saw_csls" \
                                "$K" \
                                "$SAW_SHRINK" \
                                "$TAU" \
                                "$SINKHORN_ITERS" \
                                "$SOFT_STEPS" \
                                "$SOFT_POWER"
                        done
                    done
                done
            done
        done
    done
done

python3 - "$RUN_ROOT" <<'PY'
import pandas as pd, glob, os, sys
files = glob.glob(os.path.join(sys.argv[1], "**", "inter_subject_summary.csv"), recursive=True)
df = pd.concat([pd.read_csv(f) for f in files])
avg = df.groupby('sub')[['best top1 acc', 'best top5 acc']].mean().reset_index()
avg = pd.concat([avg[avg['sub'] != 'Average'], avg[avg['sub'] == 'Average']])
avg.to_csv(os.path.join(sys.argv[1], "final_average_results.csv"), index=False)
print(f"Final average results saved to: {os.path.join(sys.argv[1], 'final_average_results.csv')}")
PY

echo "Mixup SATTC sweep completed: ${UNIFIED_CSV}"
