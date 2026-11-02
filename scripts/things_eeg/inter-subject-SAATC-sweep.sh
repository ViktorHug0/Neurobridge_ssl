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

# SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/mixup_20260330-225732/mix_raw_eeg_pairwise_linear_a0p5_seed3301}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128}"

# SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${REPO_ROOT}/results/things_eeg/inter-subjects/mixup_20260330-225732/baseline_linear_seed3302}"

HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SOFT_STEP_VALUES="${SOFT_STEP_VALUES:-8 10 12 14 16 18}"
SOFT_POWER_VALUES="${SOFT_POWER_VALUES:-1.0}"
SINKHORN_ITER_VALUES="${SINKHORN_ITER_VALUES:-1 2 5 7 10 15 20}"
TAU_VALUES="${TAU_VALUES:-0.1}"
K_VALUES="${K_VALUES:-1}"
SAW_SHRINK_VALUES="${SAW_SHRINK_VALUES:-0.85}"

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
    local sinkhorn_enabled="$5"
    local sinkhorn_tau="$6"
    local sinkhorn_iters="$7"
    local soft_enabled="$8"
    local soft_steps="$9"
    local soft_power="${10}"
    local saw_no_renorm="${11}"
    local soft_norm_inputs="${12}"

    local config_name="${SOURCE_LABEL}_$(sanitize_tag "$tag")"
    local config_run_dir="${RUN_ROOT}/${config_name}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Sweep: ${config_name}"
    echo "tag=${tag} mode=${eval_mode} k=${csls_k} saw=${saw_shrink} renorm=$((1 - saw_no_renorm)) sinkhorn=${sinkhorn_enabled}/${sinkhorn_tau}/${sinkhorn_iters} soft=${soft_enabled}/${soft_steps}/${soft_power} soft_norm=${soft_norm_inputs}"
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
            if [ "$saw_no_renorm" = "1" ]; then
                EXTRA_ARGS+=(--sattc_saw_no_renorm)
            fi
        fi
        if [ "$eval_mode" = "saw_csls" ]; then
            EXTRA_ARGS+=(--sattc_csls_k "$csls_k")
        fi
        if [ "$sinkhorn_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_sinkhorn --sattc_sinkhorn_tau "$sinkhorn_tau" --sattc_sinkhorn_iters "$sinkhorn_iters")
        fi
        if [ "$soft_enabled" = "1" ]; then
            EXTRA_ARGS+=(--sattc_soft_procrustes --sattc_soft_procrustes_steps "$soft_steps" --sattc_soft_procrustes_power "$soft_power")
            if [ "$soft_norm_inputs" = "1" ]; then
                EXTRA_ARGS+=(--sattc_soft_procrustes_normalize_inputs)
            fi
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    python3 "${REPO_ROOT}/scripts/data/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$config_name" "$config_run_dir"
}

# run_spec "exp1_k1_ablate_cw_saw_no_renorm" "saw_csls" "1" "0.85" "1" "0.08" "30" "1" "7" "1.0" "1" "0"
# run_spec "exp2_k1_ablate_cw_soft_norm_inputs" "saw_csls" "1" "0.85" "1" "0.08" "30" "1" "7" "1.0" "0" "1"

for SAW_SHRINK in ${SAW_SHRINK_VALUES}
do
    for K in ${K_VALUES}
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
                        run_spec "$(sanitize_tag "$TAG")" "saw_csls" "$K" "$SAW_SHRINK" "1" "$TAU" "$SINKHORN_ITERS" "1" "$SOFT_STEPS" "$SOFT_POWER" "0" "0"
                    done
                done
            done
        done
    done
done

# run_spec "exp7_k1_ablate_cw_steps_3_power_1p5_saw_no_renorm" "saw_csls" "1" "0.85" "1" "0.08" "30" "1" "3" "1.5" "1" "0"

echo "Targeted SAATC sweep completed: ${UNIFIED_CSV}"
