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
OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-ultimate_ablation_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/ablation_summary.csv"

# Default source run dir from the best known featdim_128 run
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128}"

HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

# Parameters for the components
SAW_SHRINK=0.85
CSLS_K=1
SINKHORN_TAU=0.1
PROCRUSTE_STEPS=10
PROCRUSTE_POWER=1.0

append_average_row() {
    local csls="$1"
    local saw="$2"
    local sinkhorn="$3"
    local procruste="$4"
    local run_dir="$5"

    python3 - "$UNIFIED_CSV" "$csls" "$saw" "$sinkhorn" "$procruste" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, csls, saw, sinkhorn, procruste, run_dir = sys.argv[1:7]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "CSLS": csls,
    "SAW": saw,
    "SINKHORN": sinkhorn,
    "PROCRUSTE": procruste,
    "top1 acc": "",
    "top5 acc": "",
}

if os.path.isfile(summary_path):
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("sub", "").strip().lower() == "average":
                row["top1 acc"] = r.get("top1 acc", "")
                row["top5 acc"] = r.get("top5 acc", "")
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

run_ablation() {
    local use_csls="$1"
    local use_saw="$2"
    local use_sinkhorn="$3"
    local use_procruste="$4"

    local label="csls${use_csls}_saw${use_saw}_sink${use_sinkhorn}_proc${use_procruste}"
    local config_run_dir="${RUN_ROOT}/${label}"
    mkdir -p "$config_run_dir"

    # Map flags to eval_mode
    local eval_mode="plain_cosine"
    if [ "$use_saw" = "1" ] && [ "$use_csls" = "1" ]; then
        eval_mode="saw_csls"
    elif [ "$use_saw" = "1" ]; then
        eval_mode="saw"
    elif [ "$use_csls" = "1" ]; then
        eval_mode="csls"
    fi

    echo "=========================================================="
    echo "Ablation Step: CSLS=$use_csls SAW=$use_saw SINKHORN=$use_sinkhorn PROCRUSTE=$use_procruste"
    echo "eval_mode=$eval_mode label=$label"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$SOURCE_RUN_DIR" "$OUTPUT_NAME")"
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${SOURCE_RUN_DIR}"
            exit 1
        fi

        EXTRA_ARGS=(
            --checkpoint_dir "$CHECKPOINT_DIR"
            --output_dir "$config_run_dir"
            --output_name "$OUTPUT_NAME"
            --eval_mode "$eval_mode"
            --test_subject_id "$SUB_ID"
            --batch_size "$BATCH_SIZE"
            --num_workers "$NUM_WORKERS"
            --device "$DEVICE"
        )

        if [ "$use_saw" = "1" ]; then
            EXTRA_ARGS+=(--sattc_saw_shrink "$SAW_SHRINK")
        fi
        if [ "$use_csls" = "1" ]; then
            EXTRA_ARGS+=(--sattc_csls_k "$CSLS_K")
        fi
        if [ "$use_sinkhorn" = "1" ]; then
            EXTRA_ARGS+=(--sattc_sinkhorn --sattc_sinkhorn_tau "$SINKHORN_TAU" --sattc_sinkhorn_iters 30)
        fi
        if [ "$use_procruste" = "1" ]; then
            EXTRA_ARGS+=(--sattc_soft_procrustes --sattc_soft_procrustes_steps "$PROCRUSTE_STEPS" --sattc_soft_procrustes_power "$PROCRUSTE_POWER")
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    # Aggregate results for this configuration
    if [ -d "$config_run_dir" ]; then
        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
        append_average_row "$use_csls" "$use_saw" "$use_sinkhorn" "$use_procruste" "$config_run_dir"
    fi
}

# --- ABLATION GRID ---

# Options: CSLS, SAW, SINKHORN, PROCRUSTE
# Iterate through all 16 combinations (2^4)

for use_csls in 0 1; do
    for use_saw in 0 1; do
        for use_sinkhorn in 0 1; do
            for use_procruste in 0 1; do
                run_ablation "$use_csls" "$use_saw" "$use_sinkhorn" "$use_procruste"
            done
        done
    done
done

echo "=========================================================="
echo "Ultimate ablation grid completed."
echo "Summary at: ${UNIFIED_CSV}"
echo "=========================================================="

# Display final table
column -t -s ',' "${UNIFIED_CSV}"
