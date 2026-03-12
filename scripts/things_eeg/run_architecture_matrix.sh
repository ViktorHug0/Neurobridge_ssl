#!/bin/bash
set -u -o pipefail

# Requested protocol:
# - 1 baseline run
# - 5 runs of Architecture A
# - 5 runs of Architecture B
#
# Each run calls scripts/things_eeg/inter-subjects.sh
# and appends one row (the Average row across subjects) to a single unified CSV.
# The unified CSV is updated after every run.

ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"./results/things_eeg/inter-subjects"}
COMPARE_TAG=${COMPARE_TAG:-"baseline_vs_A_vs_B_$(date +'%Y%m%d-%H%M%S')"}
MATRIX_OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${COMPARE_TAG}"
UNIFIED_CSV="${MATRIX_OUTPUT_DIR}/unified_run_averages.csv"

mkdir -p "${MATRIX_OUTPUT_DIR}"

# Deterministic defaults; override externally if desired.
BASELINE_SEEDS=(${BASELINE_SEEDS:-2025})
A_SEEDS=(${A_SEEDS:-2101 2102 2103 2104 2105})
B_SEEDS=(${B_SEEDS:-2201 2202 2203 2204 2205})

append_unified_row () {
    local run_index="$1"
    local config_name="$2"
    local architecture="$3"
    local seed="$4"
    local run_dir="$5"
    local status="$6"

    python3 - "$UNIFIED_CSV" "$run_index" "$config_name" "$architecture" "$seed" "$run_dir" "$status" <<'PY'
import csv
import os
import sys

unified_csv, run_index, config_name, architecture, seed, run_dir, status = sys.argv[1:8]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")

row = {
    "run_index": run_index,
    "config_name": config_name,
    "architecture": architecture,
    "seed": seed,
    "status": status,
    "run_dir": run_dir,
    "avg_top1_acc": "",
    "avg_top5_acc": "",
    "avg_best_top1_acc": "",
    "avg_best_top5_acc": "",
    "avg_best_test_loss": "",
}

if status == "ok" and os.path.isfile(summary_path):
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        average_row = None
        for r in reader:
            if r.get("sub", "").strip().lower() == "average":
                average_row = r
                break
    if average_row is not None:
        row["avg_top1_acc"] = average_row.get("top1 acc", "")
        row["avg_top5_acc"] = average_row.get("top5 acc", "")
        row["avg_best_top1_acc"] = average_row.get("best top1 acc", "")
        row["avg_best_top5_acc"] = average_row.get("best top5 acc", "")
        row["avg_best_test_loss"] = average_row.get("best test loss", "")

fieldnames = [
    "run_index",
    "config_name",
    "architecture",
    "seed",
    "status",
    "run_dir",
    "avg_top1_acc",
    "avg_top5_acc",
    "avg_best_top1_acc",
    "avg_best_top5_acc",
    "avg_best_test_loss",
]

write_header = not os.path.exists(unified_csv)
with open(unified_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one () {
    local run_index="$1"
    local config_name="$2"
    local architecture="$3"
    local seed="$4"
    local extra_args="$5"
    local ssl_pretrain_epochs="$6"
    local cl_stage2_epochs="$7"
    local stage3_epochs="$8"
    local freeze_stage2="$9"
    local subject_loss_lambda="${10}"
    local ortho_lambda="${11}"

    echo "=== Run ${run_index}/11: ${config_name} (arch=${architecture}, seed=${seed}) ==="

    ARCHITECTURE="${architecture}" \
    EXTRA_ARGS="${extra_args}" \
    SSL_PRETRAIN_EPOCHS="${ssl_pretrain_epochs}" \
    CL_STAGE2_EPOCHS="${cl_stage2_epochs}" \
    STAGE3_EPOCHS="${stage3_epochs}" \
    FREEZE_ENCODER_STAGE2="${freeze_stage2}" \
    SUBJECT_LOSS_LAMBDA="${subject_loss_lambda}" \
    ORTHO_LAMBDA="${ortho_lambda}" \
    DIAGNOSTIC_EVAL=1 \
    OUTPUT_DIR="${MATRIX_OUTPUT_DIR}" \
    SEED="${seed}" \
    bash ./scripts/things_eeg/inter-subjects.sh
    local exit_code=$?

    local latest_dir=""
    latest_dir=$(ls -1dt "${MATRIX_OUTPUT_DIR}"/*_seed"${seed}" 2>/dev/null | head -n1 || true)
    if [ -z "${latest_dir}" ]; then
        latest_dir="${MATRIX_OUTPUT_DIR}"
    fi

    if [ "${exit_code}" -eq 0 ]; then
        append_unified_row "${run_index}" "${config_name}" "${architecture}" "${seed}" "${latest_dir}" "ok"
        echo "[OK] Unified CSV updated: ${UNIFIED_CSV}"
    else
        append_unified_row "${run_index}" "${config_name}" "${architecture}" "${seed}" "${latest_dir}" "failed"
        echo "[FAIL] Unified CSV updated with failed status: ${UNIFIED_CSV}"
        return "${exit_code}"
    fi
}

run_counter=0

# 1 baseline run
for seed in "${BASELINE_SEEDS[@]}"; do
    run_counter=$((run_counter + 1))
    run_one \
        "${run_counter}" \
        "baseline" \
        "baseline" \
        "${seed}" \
        "" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0.0" \
        "0.0"
done

# 5 runs architecture A
for seed in "${A_SEEDS[@]}"; do
    run_counter=$((run_counter + 1))
    run_one \
        "${run_counter}" \
        "architecture_A" \
        "invariant_bottleneck" \
        "${seed}" \
        "--multi_positive_loss --grouped_batch_sampler --samples_per_image 10 --ssl_lambda 0.01 --ssl_projector_dim 32" \
        "10" \
        "20" \
        "10" \
        "1" \
        "0.0" \
        "0.0"
done

# 5 runs architecture B
for seed in "${B_SEEDS[@]}"; do
    run_counter=$((run_counter + 1))
    run_one \
        "${run_counter}" \
        "architecture_B" \
        "factorized" \
        "${seed}" \
        "--multi_positive_loss --grouped_batch_sampler --samples_per_image 10 --ssl_lambda 0.01 --ssl_projector_dim 32" \
        "10" \
        "20" \
        "10" \
        "0" \
        "1.0" \
        "0.01"
done

if [ "${run_counter}" -ne 11 ]; then
    echo "Warning: expected 11 runs, got ${run_counter}."
fi

echo "Comparison completed. Unified CSV: ${UNIFIED_CSV}"
