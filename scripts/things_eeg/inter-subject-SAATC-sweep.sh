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
RUN_TAG="${RUN_TAG:-sattc_final_ablation_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_sweep_summary.csv"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/mixup_20260330-225732/mix_raw_eeg_pairwise_linear_a0p5_seed3301}"
HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"
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

sanitize_tag() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "$value"
}

run_spec() {
    local tag="$1"
    local eval_mode="$2"
    local csls_k="$3"
    local saw_shrink="$4"
    local cw_enabled="$5"
    local cw_shrink="$6"
    local sinkhorn_enabled="$7"
    local sinkhorn_tau="$8"
    local sinkhorn_iters="$9"
    local soft_enabled="${10}"
    local soft_steps="${11}"
    local soft_power="${12}"

    local config_name="${SOURCE_LABEL}_$(sanitize_tag "$tag")"
    local config_run_dir="${RUN_ROOT}/${config_name}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Sweep: ${config_name}"
    echo "tag=${tag} mode=${eval_mode} k=${csls_k} saw=${saw_shrink} cw=${cw_enabled}/${cw_shrink} sinkhorn=${sinkhorn_enabled}/${sinkhorn_tau}/${sinkhorn_iters} soft=${soft_enabled}/${soft_steps}/${soft_power}"
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
        if [ "$cw_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_cw --sattc_cw_shrink "$cw_shrink")
        fi
        if [ "$sinkhorn_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_sinkhorn --sattc_sinkhorn_tau "$sinkhorn_tau" --sattc_sinkhorn_iters "$sinkhorn_iters")
        fi
        if [ "$soft_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_soft_procrustes --sattc_soft_procrustes_steps "$soft_steps" --sattc_soft_procrustes_power "$soft_power")
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$config_name" "$config_run_dir"
}

RUN_SPECS=(
    "best|saw_csls|2|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "baseline_saw_cw_csls|saw_csls|2|0.85|1|0.85|0|none|none|0|none|none"
    "ablate_soft|saw_csls|2|0.85|1|0.85|1|0.08|30|0|none|none"
    "ablate_sinkhorn|saw_csls|2|0.85|1|0.85|0|none|none|1|7|1.0"
    "ablate_csls|saw|0|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "ablate_cw|saw_csls|2|0.85|0|none|1|0.08|30|1|7|1.0"
    "ablate_saw|plain_cosine|0|none|1|0.85|1|0.08|30|1|7|1.0"
    "k1|saw_csls|1|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k3|saw_csls|3|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k4|saw_csls|4|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k5|saw_csls|5|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k6|saw_csls|6|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k7|saw_csls|7|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k8|saw_csls|8|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k9|saw_csls|9|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k10|saw_csls|10|0.85|1|0.85|1|0.08|30|1|7|1.0"
    "k1_ablate_cw|saw_csls|1|0.85|0|none|1|0.08|30|1|7|1.0"
    "k3_ablate_cw|saw_csls|3|0.85|0|none|1|0.08|30|1|7|1.0"
    "k4_ablate_cw|saw_csls|4|0.85|0|none|1|0.08|30|1|7|1.0"
    "k5_ablate_cw|saw_csls|5|0.85|0|none|1|0.08|30|1|7|1.0"
    "k6_ablate_cw|saw_csls|6|0.85|0|none|1|0.08|30|1|7|1.0"
    "k7_ablate_cw|saw_csls|7|0.85|0|none|1|0.08|30|1|7|1.0"
    "k8_ablate_cw|saw_csls|8|0.85|0|none|1|0.08|30|1|7|1.0"
    "k9_ablate_cw|saw_csls|9|0.85|0|none|1|0.08|30|1|7|1.0"
    "k10_ablate_cw|saw_csls|10|0.85|0|none|1|0.08|30|1|7|1.0"
)

for SPEC in "${RUN_SPECS[@]}"
do
    IFS='|' read -r TAG EVAL_MODE CSLS_K SAW_SHRINK CW_ENABLED CW_SHRINK SINKHORN_ENABLED SINKHORN_TAU SINKHORN_ITERS SOFT_ENABLED SOFT_STEPS SOFT_POWER <<< "$SPEC"
    run_spec "$TAG" "$EVAL_MODE" "$CSLS_K" "$SAW_SHRINK" "$CW_ENABLED" "$CW_SHRINK" "$SINKHORN_ENABLED" "$SINKHORN_TAU" "$SINKHORN_ITERS" "$SOFT_ENABLED" "$SOFT_STEPS" "$SOFT_POWER"
done

echo "Final ablation and local-optimum sweep completed: ${UNIFIED_CSV}"
