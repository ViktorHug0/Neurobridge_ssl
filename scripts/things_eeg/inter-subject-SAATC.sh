#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-sattc_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_summary.csv"

# 10 LOSO checkpoints (mixup-trained base predictor).
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/mixup_20260330-225732/mix_raw_eeg_pairwise_linear_a0p5_seed3301}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

# SAW + CW + fixed-k CSLS sweep (override CSLS_KS / SAW_SHRINKS / CW_SHRINKS).
CSLS_KS="${CSLS_KS:-2}"
SAW_SHRINKS="${SAW_SHRINKS:- 0.85}"
CW_SHRINKS="${CW_SHRINKS:-0.85}"

mkdir -p "$RUN_ROOT"

read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"
read -r -a CSLS_K_ARR <<< "$CSLS_KS"
read -r -a SAW_SHRINK_ARR <<< "$SAW_SHRINKS"
read -r -a CW_SHRINK_ARR <<< "$CW_SHRINKS"
SOURCE_LABEL="$(basename "$SOURCE_RUN_DIR")"

append_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$UNIFIED_CSV" "$config_name" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, config_name, run_dir = sys.argv[1:4]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
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

if [ ! -d "$SOURCE_RUN_DIR" ]; then
    echo "Source run directory does not exist: $SOURCE_RUN_DIR"
    exit 1
fi

for K in "${CSLS_K_ARR[@]}"
do
    for SHRINK in "${SAW_SHRINK_ARR[@]}"
    do
        for CW_SHRINK in "${CW_SHRINK_ARR[@]}"
        do
            SHRINK_TAG="${SHRINK//./p}"
            CW_TAG="${CW_SHRINK//./p}"
            CONFIG_SUFFIX="saw_cw_csls_k${K}_s${SHRINK_TAG}_cw${CW_TAG}"
            CONFIG_NAME="${SOURCE_LABEL}_${CONFIG_SUFFIX}"
            CONFIG_RUN_DIR="${RUN_ROOT}/${CONFIG_NAME}"
            mkdir -p "$CONFIG_RUN_DIR"

            echo "=========================================================="
            echo "Sweep: ${CONFIG_NAME}"
            echo "Source checkpoints: ${SOURCE_RUN_DIR}"
            echo "saw_csls + cw  k=${K}  sattc_saw_shrink=${SHRINK}  sattc_cw_shrink=${CW_SHRINK}"
            echo "=========================================================="

            for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
            do
                OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
                CHECKPOINT_DIR="$(find_checkpoint_dir "$SOURCE_RUN_DIR" "$OUTPUT_NAME")"
                if [ -z "$CHECKPOINT_DIR" ]; then
                    echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${SOURCE_RUN_DIR}"
                    exit 1
                fi

                python3 evaluate.py \
                    --checkpoint_dir "$CHECKPOINT_DIR" \
                    --output_dir "$CONFIG_RUN_DIR" \
                    --output_name "$OUTPUT_NAME" \
                    --eval_mode saw_csls \
                    --test_subject_id "$SUB_ID" \
                    --batch_size "$BATCH_SIZE" \
                    --num_workers "$NUM_WORKERS" \
                    --device "$DEVICE" \
                    --sattc_csls_k "$K" \
                    --sattc_saw_shrink "$SHRINK" \
                    --sattc_cw \
                    --sattc_cw_shrink "$CW_SHRINK"
            done

            python3 compute_avg_results.py --result_dir "$CONFIG_RUN_DIR" --output_name "inter_subject_summary.csv"
            append_average_row "$CONFIG_NAME" "$CONFIG_RUN_DIR"
        done
    done
done

echo "SAW+CW+CSLS eval sweep completed: ${UNIFIED_CSV}"
