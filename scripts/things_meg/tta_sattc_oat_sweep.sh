#!/bin/bash
# One-at-a-time (OAT) SATTC / evaluate.py TTA sweeps on fixed inter-subject checkpoints.
# This is NOT a full factorial grid: for each hyperparameter we hold every other SATTC knob
# at the reference run (saw0p65_k3_tau0p08_steps8_pow1p0_iters10), vary only that one,
# write sweep_<dim>/ CSV + one accuracy plot, then move on to the next parameter.
#
# Default ~10 trial values per axis (override with TAU_VALUES, K_VALUES, …).
# SWEEP_DIM=all runs tau → k → saw → steps → power → iters in that order.
#
# For each swept value: all held-out subjects × seeds 3300–3302, then mean(best top1/top5) across seeds.
#
# Optional fetch: FETCH_REMOTE points at the session folder that contains the three seed run dirs.
#
# Examples:
#   ./scripts/things_eeg/tta_sattc_oat_sweep.sh
#   SWEEP_DIM=tau ./scripts/things_eeg/tta_sattc_oat_sweep.sh
#   TAU_VALUES="0.06 0.08 0.1" SWEEP_DIM=tau ./scripts/things_eeg/tta_sattc_oat_sweep.sh
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DROP_SESSION="${DROP_SESSION:-things_meg_mixup_20260505-225744}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/results/things_meg/inter-subjects/${DROP_SESSION}}"
SEEDS=(3300)
CONFIG_STEM="${CONFIG_STEM:-tsconv_fd512_mixup_raw_pairwise_linear_a0p5_seed}"

HELD_OUT_SUBJECTS="${HELD_OUT_SUBJECTS:-1 2 3 4}"

# Reference SATTC — held fixed while sweeping any single dimension below
BASE_K="${BASE_K:-3}"
BASE_SAW="${BASE_SAW:-0.94}"
BASE_TAU="${BASE_TAU:-0.1}"
BASE_SOFT_STEPS="${BASE_SOFT_STEPS:-16}"
BASE_SOFT_POWER="${BASE_SOFT_POWER:-1.2}"
BASE_SINKHORN_ITERS="${BASE_SINKHORN_ITERS:-12}"

# all | tau | k | saw | steps | power | iters
SWEEP_DIM="${SWEEP_DIM:-all}"

OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/results/things_meg/inter-subjects}"
RUN_STAMP="${RUN_STAMP:-$(date +'%Y%m%d-%H%M%S')}"
SESSION_ROOT="${OUTPUT_ROOT}/sattc_oat_${RUN_STAMP}"

# ~10 values per axis by default (full grid never explored — OAT only)
TAU_VALUES="${TAU_VALUES:-0.05 0.07 0.08 0.10 0.12 0.15 0.18 0.20 0.25}"
K_VALUES="${K_VALUES:-1 3 5 10}"
SAW_VALUES="${SAW_VALUES:-0.75 0.80 0.85 0.90 0.95 0.96 0.97 0.98 0.99}"
SOFT_STEP_VALUES="${SOFT_STEP_VALUES:-2 4 6 8 10 12 14 16}"
SOFT_POWER_VALUES="${SOFT_POWER_VALUES:-0.6 0.7 0.8 0.9 1.0 1.1 1.25}"
SINKHORN_ITER_VALUES="${SINKHORN_ITER_VALUES:-2 4 6 8 10 15 20}"

# Set per-dimension RUN_ROOT / SUMMARY_CSV (used by run_eval_for_value, run_sweep, plot_summary)
RUN_ROOT=""
SUMMARY_CSV=""

sanitize_tag() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "$value"
}

find_checkpoint_dir() {
    local source_run_dir="$1"
    local output_name="$2"
    ls -td "${source_run_dir}"/*-"${output_name}" 2>/dev/null | head -n 1
}

fetch_checkpoints() {
    if [ -z "${FETCH_REMOTE:-}" ]; then
        return 0
    fi
    echo "Fetching checkpoints from ${FETCH_REMOTE} into ${SOURCE_RUN_ROOT}"
    mkdir -p "$SOURCE_RUN_ROOT"
    for SEED in "${SEEDS[@]}"; do
        local name="${CONFIG_STEM}${SEED}"
        rsync -aL "${FETCH_REMOTE}/${name}/" "${SOURCE_RUN_ROOT}/${name}/"
    done
}

set_outputs_for_dim() {
    local d="$1"
    RUN_ROOT="${SESSION_ROOT}/sweep_${d}"
    SUMMARY_CSV="${RUN_ROOT}/oat_${d}_mean_across_seeds.csv"
    mkdir -p "$RUN_ROOT"
}

run_eval_for_value() {
    local sweep_val="$1"
    local k="$2"
    local saw="$3"
    local tau="$4"
    local soft_steps="$5"
    local soft_power="$6"
    local sink_iters="$7"

    local tag="saw${saw}_k${k}_tau${tau}_steps${soft_steps}_pow${soft_power}_iters${sink_iters}"
    local val_dir
    val_dir="$(sanitize_tag "$sweep_val")"
    read -r -a HELD_OUT_SUBJECT_ARR <<< "$HELD_OUT_SUBJECTS"

    for SEED in "${SEEDS[@]}"; do
        local source_run_dir="${SOURCE_RUN_ROOT}/${CONFIG_STEM}${SEED}"
        if [ ! -d "$source_run_dir" ]; then
            echo "Missing source dir: $source_run_dir"
            exit 1
        fi
        local config_run_dir="${RUN_ROOT}/dim_${SWEEP_DIM}/${val_dir}/seed_${SEED}"
        mkdir -p "$config_run_dir"

        echo "=== seed=${SEED} ${SWEEP_DIM}=${sweep_val} (full tag ${tag}) ==="
        for SUB_ID in "${HELD_OUT_SUBJECT_ARR[@]}"; do
            local output_name
            output_name=$(printf "sub-%02d" "$SUB_ID")
            local checkpoint_dir
            checkpoint_dir="$(find_checkpoint_dir "$source_run_dir" "$output_name")"
            if [ -z "$checkpoint_dir" ]; then
                echo "No checkpoint for ${output_name} in ${source_run_dir}"
                exit 1
            fi
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python3 "${REPO_ROOT}/evaluate.py" \
                --checkpoint_dir "$checkpoint_dir" \
                --output_dir "$config_run_dir" \
                --output_name "$output_name" \
                --eval_mode saw_csls \
                --test_subject_id "$SUB_ID" \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --device "$DEVICE" \
                --sattc_saw_shrink "$saw" \
                --sattc_csls_k "$k" \
                --sattc_sinkhorn \
                --sattc_sinkhorn_tau "$tau" \
                --sattc_sinkhorn_iters "$sink_iters" \
                --sattc_soft_procrustes \
                --sattc_soft_procrustes_steps "$soft_steps" \
                --sattc_soft_procrustes_power "$soft_power"
        done
        python3 "${REPO_ROOT}/compute_avg_results.py" --result_dir "$config_run_dir" --output_name "inter_subject_summary.csv"
    done

    python3 - "$SUMMARY_CSV" "$SWEEP_DIM" "$sweep_val" "${RUN_ROOT}/dim_${SWEEP_DIM}/${val_dir}" "${SEEDS[*]}" <<'PY'
import csv
import os
import sys

summary_append, dim, sweep_val, group_dir, seeds_s = sys.argv[1:6]
seeds = [int(x) for x in seeds_s.split()]
rows = []
for s in seeds:
    path = os.path.join(group_dir, f"seed_{s}", "inter_subject_summary.csv")
    got = False
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("sub") or "").strip().lower() == "average":
                b1 = float(r["best top1 acc"])
                b5 = float(r["best top5 acc"])
                rows.append((s, b1, b5))
                got = True
                break
    if not got:
        raise SystemExit(f"Missing Average row in {path}")

mean1 = sum(b1 for _, b1, _ in rows) / len(rows)
mean5 = sum(b5 for _, _, b5 in rows) / len(rows)

write_header = not os.path.isfile(summary_append)
with open(summary_append, "a", newline="") as out:
    w = csv.writer(out)
    if write_header:
        w.writerow([dim, "mean_best_top1", "mean_best_top5", "per_seed_best_top1", "per_seed_best_top5"])
    w.writerow([sweep_val, f"{mean1:.4f}", f"{mean5:.4f}", " ".join(f"{b1:.2f}" for _, b1, _ in rows), " ".join(f"{b5:.2f}" for _, _, b5 in rows)])
print(f"{dim}={sweep_val} mean_best_top1={mean1:.2f} mean_best_top5={mean5:.2f}")
PY
}

run_sweep() {
    case "$SWEEP_DIM" in
        tau)
            for v in $TAU_VALUES; do
                run_eval_for_value "$v" "$BASE_K" "$BASE_SAW" "$v" "$BASE_SOFT_STEPS" "$BASE_SOFT_POWER" "$BASE_SINKHORN_ITERS"
            done
            ;;
        k)
            for v in $K_VALUES; do
                run_eval_for_value "$v" "$v" "$BASE_SAW" "$BASE_TAU" "$BASE_SOFT_STEPS" "$BASE_SOFT_POWER" "$BASE_SINKHORN_ITERS"
            done
            ;;
        saw)
            for v in $SAW_VALUES; do
                run_eval_for_value "$v" "$BASE_K" "$v" "$BASE_TAU" "$BASE_SOFT_STEPS" "$BASE_SOFT_POWER" "$BASE_SINKHORN_ITERS"
            done
            ;;
        steps)
            for v in $SOFT_STEP_VALUES; do
                run_eval_for_value "$v" "$BASE_K" "$BASE_SAW" "$BASE_TAU" "$v" "$BASE_SOFT_POWER" "$BASE_SINKHORN_ITERS"
            done
            ;;
        power)
            for v in $SOFT_POWER_VALUES; do
                run_eval_for_value "$v" "$BASE_K" "$BASE_SAW" "$BASE_TAU" "$BASE_SOFT_STEPS" "$v" "$BASE_SINKHORN_ITERS"
            done
            ;;
        iters)
            for v in $SINKHORN_ITER_VALUES; do
                run_eval_for_value "$v" "$BASE_K" "$BASE_SAW" "$BASE_TAU" "$BASE_SOFT_STEPS" "$BASE_SOFT_POWER" "$v"
            done
            ;;
        *)
            echo "Unknown SWEEP_DIM=$SWEEP_DIM (use all|tau|k|saw|steps|power|iters)"
            exit 1
            ;;
    esac
}

plot_summary() {
    python3 - "$SUMMARY_CSV" "${RUN_ROOT}/oat_${SWEEP_DIM}_plot.png" "$SWEEP_DIM" \
        "$BASE_K" "$BASE_SAW" "$BASE_TAU" "$BASE_SOFT_STEPS" "$BASE_SOFT_POWER" "$BASE_SINKHORN_ITERS" <<'PY'
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

csv_path, png_path, dim, bk, bs, bt, bst, bp, bi = sys.argv[1:10]
df = pd.read_csv(csv_path)
xcol = df.columns[0]
df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
df = df.sort_values(by=xcol)
xs = df[xcol].astype(float)
y1 = df["mean_best_top1"].astype(float)
y5 = df["mean_best_top5"].astype(float)
base = (
        f"OAT: only '{dim}' varies; others fixed at "
        f"k={bk} saw={bs} tau={bt} steps={bst} pow={bp} iters={bi}"
    )
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, y1, "o-", label="mean best top-1 (over seeds)")
ax.plot(xs, y5, "s-", label="mean best top-5 (over seeds)")
ax.set_xlabel(dim)
ax.set_ylabel("accuracy (%)")
ax.set_title(f"SATTC one-at-a-time sweep: {dim}\n{base}")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(png_path, dpi=150)
print(f"Wrote {png_path}")
PY
}

run_one_dimension() {
    local d="$1"
    SWEEP_DIM="$d"
    set_outputs_for_dim "$d"
    echo "=========================================================="
    echo "OAT sweep: varying ONLY ${d} (all other SATTC params fixed to BASE_* reference)."
    echo "Not a grid — single axis → CSV + plot → next parameter."
    echo "Output: ${RUN_ROOT}"
    echo "=========================================================="
    run_sweep
    plot_summary
    echo "Done ${d}: ${SUMMARY_CSV} + ${RUN_ROOT}/oat_${d}_plot.png"
}

fetch_checkpoints

mkdir -p "$SESSION_ROOT"
echo "Session: ${SESSION_ROOT}"
echo "Source checkpoints: ${SOURCE_RUN_ROOT}"
echo "Design: one-at-a-time SATTC sweeps (reference BASE_K=${BASE_K} BASE_SAW=${BASE_SAW} BASE_TAU=${BASE_TAU} BASE_SOFT_STEPS=${BASE_SOFT_STEPS} BASE_SOFT_POWER=${BASE_SOFT_POWER} BASE_SINKHORN_ITERS=${BASE_SINKHORN_ITERS})."

if [ "$SWEEP_DIM" = "all" ]; then
    for d in tau k saw steps power iters; do
        run_one_dimension "$d"
    done
    echo "All sweeps finished under ${SESSION_ROOT}"
else
    run_one_dimension "$SWEEP_DIM"
fi
