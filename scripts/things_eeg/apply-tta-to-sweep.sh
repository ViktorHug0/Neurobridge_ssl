#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# The source sweep directory provided by the user
SOURCE_SWEEP_ROOT="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/eeg_encoder_architecture_sweep_mixup_20260429-221100"

# Output directory for the TTA results
OUTPUT_ROOT="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/tta_sweep_20260503-152811"
UNIFIED_CSV="${OUTPUT_ROOT}/tta_sweep_summary.csv"

# TTA Configuration (SAW + CSLS + Sinkhorn + Soft Procrustes)
EVAL_MODE="saw_csls"
SAW_SHRINK="0.94"
CSLS_K="3"
SINKHORN_TAU="0.1"
SINKHORN_ITERS="12"
SOFT_STEPS="16"
SOFT_POWER="1.2"

HELD_OUT_SUBJECTS="1 2 3 4 5 6 7 8 9 10"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

mkdir -p "$OUTPUT_ROOT"

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

# Find all run directories (excluding the summary CSV)
shopt -s nullglob
RUN_DIRS=("$SOURCE_SWEEP_ROOT"/*EEGConformer*/)
shopt -u nullglob

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "No run directories found in $SOURCE_SWEEP_ROOT"
    exit 1
fi

for SOURCE_RUN_DIR in "${RUN_DIRS[@]}"
do
    SOURCE_LABEL="$(basename "$SOURCE_RUN_DIR")"
    CONFIG_NAME="${SOURCE_LABEL}_tta"
    CONFIG_RUN_DIR="${OUTPUT_ROOT}/${CONFIG_NAME}"
    mkdir -p "$CONFIG_RUN_DIR"

    echo "=========================================================="
    echo "Applying TTA to: ${SOURCE_LABEL}"
    echo "Output: ${CONFIG_RUN_DIR}"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$SOURCE_RUN_DIR" "$OUTPUT_NAME")"
        
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Warning: Could not find checkpoint directory for ${OUTPUT_NAME} in ${SOURCE_RUN_DIR}. Skipping subject."
            continue
        fi

        python3 "${REPO_ROOT}/evaluate.py" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$CONFIG_RUN_DIR" \
            --output_name "$OUTPUT_NAME" \
            --eval_mode "$EVAL_MODE" \
            --test_subject_id "$SUB_ID" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --device "$DEVICE" \
            --sattc_saw_shrink "$SAW_SHRINK" \
            --sattc_csls_k "$CSLS_K" \
            --sattc_sinkhorn \
            --sattc_sinkhorn_tau "$SINKHORN_TAU" \
            --sattc_sinkhorn_iters "$SINKHORN_ITERS" \
            --sattc_soft_procrustes \
            --sattc_soft_procrustes_steps "$SOFT_STEPS" \
            --sattc_soft_procrustes_power "$SOFT_POWER"
    done

    # Compute average results for this run
    if [ "$(ls -A "$CONFIG_RUN_DIR")" ]; then
        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
        append_average_row "$SOURCE_LABEL" "$CONFIG_NAME" "$CONFIG_RUN_DIR"
    fi
done

echo "TTA sweep completed. Summary: ${UNIFIED_CSV}"
