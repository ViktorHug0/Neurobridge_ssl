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
RUN_TAG="sattc_mixup_ablation_sweep_$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sattc_mixup_ablation_summary.csv"

# Updated TTA parameters from user: "saw0p93_k1_tau0p09_steps18_pow1p0_iters14"
BASE_SAW_SHRINK="0.94"
BASE_CSLS_K="3"
BASE_SINKHORN_TAU="0.1"
BASE_SINKHORN_ITERS="12"
BASE_SOFT_STEPS="16"
BASE_SOFT_POWER="1.2"
BASE_SOFT_ASSIGNMENT_TOPK="5"

HELD_OUT_SUBJECTS="1 2 3 4 5 6 7 8 9 10"
mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

append_average_row() {
    local variant="$1"
    local mixup="$2"
    local description="$3"
    local eval_mode="$4"
    local use_saw="$5"
    local use_csls="$6"
    local use_sinkhorn="$7"
    local use_soft="$8"
    local run_dir="$9"
    python3 - "$UNIFIED_CSV" "$variant" "$mixup" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$run_dir" <<'PY'
import csv
import os
import sys

(
    out_csv,
    variant,
    mixup,
    description,
    eval_mode,
    use_saw,
    use_csls,
    use_sinkhorn,
    use_soft,
    run_dir,
) = sys.argv[1:11]

summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "variant": variant,
    "mixup": mixup,
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
}

if os.path.isfile(summary_path):
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("sub", "").strip().lower() == "average":
                for key in ("top1 acc", "top5 acc", "best top1 acc", "best top5 acc"):
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
    local mixup="$2"
    local description="$3"
    local use_saw="$4"
    local use_csls="$5"
    local use_sinkhorn="$6"
    local use_soft="$7"
    local source_run_root="$8"
    local source_config="$9"

    local source_run_dir="${source_run_root}/${source_config}"
    
    local eval_mode="plain_cosine"
    if [ "$use_saw" = "1" ] && [ "$use_csls" = "1" ]; then
        eval_mode="saw_csls"
    elif [ "$use_saw" = "1" ]; then
        eval_mode="saw"
    elif [ "$use_csls" = "1" ]; then
        eval_mode="csls"
    fi

    local config_run_dir="${RUN_ROOT}/${mixup}_${variant}_${source_config}"
    mkdir -p "$config_run_dir"

    echo "=========================================================="
    echo "Variant: ${variant} | Mixup: ${mixup} | Seed: ${source_config}"
    echo "eval_mode=${eval_mode} saw=${use_saw} csls=${use_csls} sinkhorn=${use_sinkhorn} soft=${use_soft}"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"
    do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$source_run_dir" "$OUTPUT_NAME")"
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${source_run_dir}"
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
        fi

        python3 "${REPO_ROOT}/evaluate.py" "${EXTRA_ARGS[@]}"
    done

    python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    append_average_row "$variant" "$mixup" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$config_run_dir"
}

# Source roots
MIXUP_ROOT="${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741"
NOMIXUP_ROOT="${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-170207"

# All training checkpoints used for this ablation (must exist under both MIXUP_ROOT and NOMIXUP_ROOT).
SEEDS=(
    "param_k30_pool51_do050_featdim512_seed3300"
    "param_k30_pool51_do050_featdim512_seed3301"
    "param_k30_pool51_do050_featdim512_seed3302"
)

echo "Running ablation across ${#SEEDS[@]} seeds (3300, 3301, 3302); per-seed rows -> ${UNIFIED_CSV}, means -> ${UNIFIED_CSV%.csv}_aggregated.csv"

for SEED in "${SEEDS[@]}"; do
    # Mixup Enabled
    run_variant "baseline_best_sattc" "enabled" "Full SATTC" 1 1 1 1 "$MIXUP_ROOT" "$SEED"
    run_variant "ablate_whitening" "enabled" "No Whitening" 0 1 1 1 "$MIXUP_ROOT" "$SEED"
    run_variant "ablate_csls" "enabled" "No CSLS" 1 0 1 1 "$MIXUP_ROOT" "$SEED"
    run_variant "ablate_structural" "enabled" "No Structural (Sinkhorn+SoftProc)" 1 1 0 0 "$MIXUP_ROOT" "$SEED"
    run_variant "plain_cosine" "enabled" "No TTA" 0 0 0 0 "$MIXUP_ROOT" "$SEED"

    # Mixup Disabled
    run_variant "baseline_best_sattc" "disabled" "Full SATTC" 1 1 1 1 "$NOMIXUP_ROOT" "$SEED"
    run_variant "ablate_whitening" "disabled" "No Whitening" 0 1 1 1 "$NOMIXUP_ROOT" "$SEED"
    run_variant "ablate_csls" "disabled" "No CSLS" 1 0 1 1 "$NOMIXUP_ROOT" "$SEED"
    run_variant "ablate_structural" "disabled" "No Structural (Sinkhorn+SoftProc)" 1 1 0 0 "$NOMIXUP_ROOT" "$SEED"
    run_variant "plain_cosine" "disabled" "No TTA" 0 0 0 0 "$NOMIXUP_ROOT" "$SEED"
done

# Final aggregation across seeds
python3 - "$UNIFIED_CSV" <<'PY'
import csv
import pandas as pd
import sys

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# Group by variant and mixup to average across seeds
numeric_cols = ["top1 acc", "top5 acc", "best top1 acc", "best top5 acc"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

summary = df.groupby(["mixup", "variant"])[numeric_cols].mean().reset_index()
summary.to_csv(csv_path.replace(".csv", "_aggregated.csv"), index=False)
PY

echo "Experiment completed. Summary at: ${UNIFIED_CSV}"
echo "Seed-aggregated summary at: ${UNIFIED_CSV%.csv}_aggregated.csv"
