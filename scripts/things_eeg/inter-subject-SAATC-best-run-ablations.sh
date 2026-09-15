#!/bin/bash
set -euo pipefail
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
RUN_TAG="${RUN_TAG:-sattc_best_run_ablations_20260427-170609}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_ablation_summary.csv"

# Best-performing source model from the dropout sweep.
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260426-015522}"
SOURCE_CONFIG="${SOURCE_CONFIG:-param_k30_pool51_do050_featdim64_seed3300}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${SOURCE_RUN_ROOT}/${SOURCE_CONFIG}}"

HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

# Exact SATTC settings from sattc_final_paper_sweep_run_20260426-211731 row 5.
BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.8}"
BASE_CSLS_K="${BASE_CSLS_K:-1}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-4}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.0}"
BASE_SOFT_ASSIGNMENT_TOPK="${BASE_SOFT_ASSIGNMENT_TOPK:-5}"

mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

if [ ! -d "$SOURCE_RUN_DIR" ]; then
    echo "Source run directory does not exist: $SOURCE_RUN_DIR"
    exit 1
fi

append_average_row() {
    local variant="$1"
    local description="$2"
    local eval_mode="$3"
    local use_saw="$4"
    local use_csls="$5"
    local use_sinkhorn="$6"
    local use_soft="$7"
    local run_dir="$8"
    python3 - "$UNIFIED_CSV" "$variant" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$run_dir" <<'PY'
import csv
import os
import sys

(
    out_csv,
    variant,
    description,
    eval_mode,
    use_saw,
    use_csls,
    use_sinkhorn,
    use_soft,
    run_dir,
) = sys.argv[1:10]

summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "variant": variant,
    "description": description,
    "eval_mode": eval_mode,
    "use_saw": use_saw,
    "use_csls": use_csls,
    "use_sinkhorn": use_sinkhorn,
    "use_soft_procrustes": use_soft,
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
                for key in ("top1 acc", "top5 acc", "best top1 acc", "best top5 acc", "best test loss", "best epoch"):
                    row[key] = r.get(key, "")
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

run_variant() {
    local variant="$1"
    local description="$2"
    local use_saw="$3"
    local use_csls="$4"
    local use_sinkhorn="$5"
    local use_soft="$6"
    local soft_assignment_method="${7:-sinkhorn}"
    local soft_assignment_topk="${8:-$BASE_SOFT_ASSIGNMENT_TOPK}"

    local eval_mode="plain_cosine"
    if [ "$use_saw" = "1" ] && [ "$use_csls" = "1" ]; then
        eval_mode="saw_csls"
    elif [ "$use_saw" = "1" ]; then
        eval_mode="saw"
    elif [ "$use_csls" = "1" ]; then
        eval_mode="csls"
    fi

    local config_run_dir="${RUN_ROOT}/${variant}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Variant: ${variant}"
    echo "Description: ${description}"
    echo "eval_mode=${eval_mode} saw=${use_saw} csls=${use_csls} sinkhorn=${use_sinkhorn} soft=${use_soft} assignment=${soft_assignment_method}"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$SOURCE_RUN_DIR" "$OUTPUT_NAME")"
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${SOURCE_RUN_DIR}"
            exit 1
        fi

        if compgen -G "${config_run_dir}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
            continue
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
            EXTRA_ARGS+=(--sattc_saw_shrink "$BASE_SAW_SHRINK")
        fi
        if [ "$use_csls" = "1" ]; then
            EXTRA_ARGS+=(--sattc_csls_k "$BASE_CSLS_K")
        fi
        if [ "$use_sinkhorn" = "1" ]; then
            EXTRA_ARGS+=(--sattc_sinkhorn --sattc_sinkhorn_tau "$BASE_SINKHORN_TAU" --sattc_sinkhorn_iters "$BASE_SINKHORN_ITERS")
        fi
        if [ "$use_soft" = "1" ]; then
            EXTRA_ARGS+=(--sattc_soft_procrustes --sattc_soft_procrustes_steps "$BASE_SOFT_STEPS" --sattc_soft_procrustes_power "$BASE_SOFT_POWER")
            EXTRA_ARGS+=(--sattc_soft_procrustes_assignment "$soft_assignment_method")
            if [ "$soft_assignment_method" = "topk" ]; then
                EXTRA_ARGS+=(--sattc_soft_procrustes_assignment_topk "$soft_assignment_topk")
            fi
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$variant" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$config_run_dir"
}

run_variant "baseline_best_sattc" "Exact best SATTC row: SAW + CSLS + soft Procrustes + final Sinkhorn." 1 1 1 1
run_variant "ablate_subject_side_whitening" "Disable subject-side whitening only; keep the rest of the best TTA pipeline." 0 1 1 1
run_variant "ablate_csls" "Disable CSLS only; keep SAW and structural refinement." 1 0 1 1
run_variant "ablate_soft_procrustes" "Disable soft Procrustes only; keep SAW, CSLS, and final Sinkhorn." 1 1 1 0
run_variant "ablate_sinkhorn_regularization" "Disable final Sinkhorn regularization only; keep SAW, CSLS, and soft Procrustes." 1 1 0 1

# Useful references beyond one-at-a-time ablations.
run_variant "ablate_structural_refinement" "Disable both soft Procrustes and final Sinkhorn; leaves SAW + fixed-k CSLS." 1 1 0 0
run_variant "plain_cosine_reference" "Disable the full SATTC TTA stack for a no-TTA cosine reference." 0 0 0 0
run_variant "no_sinkhorn_softmax_assignment" "Disable all Sinkhorn usage; use SAW + CSLS with iterative soft Procrustes driven by row-wise softmax assignments." 1 1 0 1 "softmax"
run_variant "no_sinkhorn_topk_assignment" "Disable all Sinkhorn usage; use SAW + CSLS with iterative soft Procrustes driven by top-k row-normalized assignments." 1 1 0 1 "topk" "$BASE_SOFT_ASSIGNMENT_TOPK"
run_variant "no_sinkhorn_argmax_assignment" "Disable all Sinkhorn usage; use SAW + CSLS with iterative soft Procrustes driven by hard argmax assignments." 1 1 0 1 "argmax"

echo "SATTC best-run ablations completed: ${UNIFIED_CSV}"
