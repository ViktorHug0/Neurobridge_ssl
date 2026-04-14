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

run_ablation() {
    local label="$1"
    local eval_mode="$2"
    local csls_k="$3"
    local saw_shrink="$4"
    local sinkhorn_enabled="$5"
    local soft_procrustes_enabled="$6"
    local soft_steps="$7"
    local soft_power="$8"
    local sinkhorn_tau="$9"

    local config_run_dir="${RUN_ROOT}/${label}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Ablation Step: ${label}"
    echo "eval_mode=${eval_mode} k=${csls_k} saw=${saw_shrink} sinkhorn=${sinkhorn_enabled} soft=${soft_procrustes_enabled}/${soft_steps}/${soft_power} tau=${sinkhorn_tau}"
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

        if [ "$eval_mode" = "saw" ] || [ "$eval_mode" = "saw_csls" ]; then
            EXTRA_ARGS+=(--sattc_saw_shrink "$saw_shrink")
        fi
        if [ "$eval_mode" = "saw_csls" ]; then
            EXTRA_ARGS+=(--sattc_csls_k "$csls_k")
        fi
        if [ "$sinkhorn_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_sinkhorn --sattc_sinkhorn_tau "$sinkhorn_tau" --sattc_sinkhorn_iters 30)
        fi
        if [ "$soft_procrustes_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_soft_procrustes --sattc_soft_procrustes_steps "$soft_steps" --sattc_soft_procrustes_power "$soft_power")
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$label" "$config_run_dir"
}

# --- ABLATION MATRIX ---

# 1. Full Best Run (SAATC: SAW + CSLS + Soft-Procrustes + Final Sinkhorn)
run_ablation "01_full_best" "saw_csls" 1 0.85 1 1 10 1.0 0.1

# 2. Disable Soft-Procrustes Refinement
run_ablation "02_no_soft_procrustes" "saw_csls" 1 0.85 1 0 0 1.0 0.1

# 3. Disable Final Sinkhorn Normalization
run_ablation "03_no_final_sinkhorn" "saw_csls" 1 0.85 0 1 10 1.0 0.1

# 4. Disable Both Test-Time Refinements (Keep SAW + CSLS)
run_ablation "04_no_refinement" "saw_csls" 1 0.85 0 0 0 1.0 0.1

# 5. Disable CSLS (Keep SAW + Refinements)
run_ablation "05_no_csls" "saw" 1 0.85 1 1 10 1.0 0.1

# 6. Disable CSLS and Refinements (Just SAW)
run_ablation "06_just_saw" "saw" 1 0.85 0 0 0 1.0 0.1

# 7. Baseline (Plain Cosine - No SAW, No CSLS, No Refinements)
run_ablation "07_baseline" "plain_cosine" 1 1.0 0 0 0 1.0 0.1

echo "=========================================================="
echo "Ultimate ablation completed."
echo "Summary at: ${UNIFIED_CSV}"
echo "=========================================================="
cat "${UNIFIED_CSV}"
