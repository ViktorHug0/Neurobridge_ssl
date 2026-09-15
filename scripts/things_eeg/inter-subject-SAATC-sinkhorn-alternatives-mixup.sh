#!/bin/bash
# Sinkhorn vs softmax / top-k / argmax assignment (SubjectMix checkpoints + TTA from reproduce_ablation_with_mixup.sh).
# Output: per-seed rows + *_aggregated.csv (mean over seeds) for a 4-row LaTeX table.
#
# Resume: per-subject result.csv under RUN_ROOT is skipped if present. Summary rows include
# `seed`; aggregation keeps the last row per (mixup, variant, seed) so full re-runs do not
# inflate the 3-seed mean.
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
RUN_TAG="${RUN_TAG:-sattc_sinkhorn_alt_mixup_${RUN_TAG_SUFFIX:-$(date +%Y%m%d-%H%M%S)}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/sinkhorn_alternatives_mixup_summary.csv"

# Match scripts/things_eeg/reproduce_ablation_with_mixup.sh (SubjectMix + TTA)
MIXUP_ROOT="${MIXUP_ROOT:-${REPO_ROOT}/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741}"
BASE_SAW_SHRINK="${BASE_SAW_SHRINK:-0.94}"
BASE_CSLS_K="${BASE_CSLS_K:-3}"
BASE_SINKHORN_TAU="${BASE_SINKHORN_TAU:-0.1}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"
BASE_SOFT_ASSIGNMENT_TOPK="${BASE_SOFT_ASSIGNMENT_TOPK:-5}"

SEEDS=(
    "param_k30_pool51_do050_featdim512_seed3300"
    "param_k30_pool51_do050_featdim512_seed3301"
    "param_k30_pool51_do050_featdim512_seed3302"
)

HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
mkdir -p "$RUN_ROOT"
read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

for SEED_DIR in "${SEEDS[@]}"; do
    if [ ! -d "${MIXUP_ROOT}/${SEED_DIR}" ]; then
        echo "Missing mixup checkpoints: ${MIXUP_ROOT}/${SEED_DIR}" >&2
        exit 1
    fi
done

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
    local seed_cfg="${10:-}"
    python3 - "$UNIFIED_CSV" "$variant" "$mixup" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$run_dir" "$seed_cfg" <<'PY'
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
    seed_cfg,
) = sys.argv[1:12]

summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "variant": variant,
    "mixup": mixup,
    "seed": seed_cfg,
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
    local soft_assignment_method="${10:-sinkhorn}"
    local soft_assignment_topk="${11:-$BASE_SOFT_ASSIGNMENT_TOPK}"

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
    echo "Variant: ${variant} | Mixup: ${mixup} | Seed: ${source_config} | assignment=${soft_assignment_method}"
    echo "eval_mode=${eval_mode} saw=${use_saw} csls=${use_csls} sinkhorn=${use_sinkhorn} soft=${use_soft}"
    echo "=========================================================="

    for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"; do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        CHECKPOINT_DIR="$(find_checkpoint_dir "$source_run_dir" "$OUTPUT_NAME")"
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "Could not find checkpoint directory for ${OUTPUT_NAME} in ${source_run_dir}"
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
    append_average_row "$variant" "$mixup" "$description" "$eval_mode" "$use_saw" "$use_csls" "$use_sinkhorn" "$use_soft" "$config_run_dir" "$source_config"
}

echo "Sinkhorn-alternative sweep (SubjectMix): ${RUN_ROOT}"

for SEED in "${SEEDS[@]}"; do
    run_variant "baseline_best_sattc" "enabled" "Full SATTC (Sinkhorn assignment + final Sinkhorn)" 1 1 1 1 "$MIXUP_ROOT" "$SEED" "sinkhorn"
    run_variant "no_sinkhorn_softmax_assignment" "enabled" "Softmax assignment; no final Sinkhorn" 1 1 0 1 "$MIXUP_ROOT" "$SEED" "softmax"
    run_variant "no_sinkhorn_topk_assignment" "enabled" "Top-k softmax assignment; no final Sinkhorn" 1 1 0 1 "$MIXUP_ROOT" "$SEED" "topk" "$BASE_SOFT_ASSIGNMENT_TOPK"
    run_variant "no_sinkhorn_argmax_assignment" "enabled" "Argmax assignment; no final Sinkhorn" 1 1 0 1 "$MIXUP_ROOT" "$SEED" "argmax"
done

python3 - "$UNIFIED_CSV" <<'PY'
import pandas as pd
import sys

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
df = df[df["mixup"].astype(str).str.lower() == "enabled"]
df = df.sort_values("seed").drop_duplicates(subset=["mixup", "variant", "seed"], keep="last")
numeric_cols = ["top1 acc", "top5 acc", "best top1 acc", "best top5 acc"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
summary = df.groupby(["mixup", "variant"], as_index=False)[numeric_cols].mean()
order = [
    "baseline_best_sattc",
    "no_sinkhorn_softmax_assignment",
    "no_sinkhorn_topk_assignment",
    "no_sinkhorn_argmax_assignment",
]
summary["_k"] = summary["variant"].map({v: i for i, v in enumerate(order)}).fillna(999)
summary = summary.sort_values("_k").drop(columns="_k")
summary.to_csv(csv_path.replace(".csv", "_aggregated.csv"), index=False)
PY

echo "Done: ${UNIFIED_CSV}"
echo "Aggregated (3-seed mean): ${UNIFIED_CSV%.csv}_aggregated.csv"
