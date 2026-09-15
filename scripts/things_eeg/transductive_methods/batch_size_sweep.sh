#!/usr/bin/env bash
#
# Test-batch-size sweep wrapper around full_benchmark_suite.sh.
#
# Repeatedly runs the full benchmark suite (every method, each in its tuned best
# config) on an eval set built from the left-out subject's TRAIN recordings
# (EVAL_SPLIT=train, EEG averaged over the 4 reps/image), varying the bijective
# test batch size. For each batch size N the suite is run with SUBSAMPLE=N
# (n_images = n_queries = N). When all sizes are done, a single summary plot is
# drawn: batch size on x, top-1 accuracy on y, one distinctly-coloured line per
# method.
#
# Config (env vars):
#   BATCH_SIZES="50 100 ..."  batch sizes to sweep (default: 50 100 150 200 250 300 500 1000)
#   DEVICE=cuda:0             passed through to the suite
#   SOURCE_RUN_DIR=<dir>      LOSO checkpoint param dir (passed through)
#
# Example:
#   bash scripts/things_eeg/transductive_methods/batch_size_sweep.sh
#   BATCH_SIZES="100 200 400" bash scripts/things_eeg/transductive_methods/batch_size_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

BATCH_SIZES="${BATCH_SIZES:-50 100 150 200 250 300 500 1000}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
DEVICE="${DEVICE:-cuda:0}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-results/batch_size_sweep/${STAMP}}"
mkdir -p "${RUN_DIR}"

echo "Batch sizes : ${BATCH_SIZES}"
echo "Eval split  : ${EVAL_SPLIT}"
echo "Run dir     : ${RUN_DIR}"
echo "Device      : ${DEVICE}"

for N in ${BATCH_SIZES}; do
    OUT="${RUN_DIR}/N${N}"
    echo "=== batch size N=${N} -> ${OUT} ==="
    EVAL_SPLIT="${EVAL_SPLIT}" \
    SUBSAMPLE="${N}" \
    DEVICE="${DEVICE}" \
    OUTPUT_DIR="${OUT}" \
    ${SOURCE_RUN_DIR:+SOURCE_RUN_DIR="${SOURCE_RUN_DIR}"} \
        bash "${SCRIPT_DIR}/full_benchmark_suite.sh"
done

# ----------------------------------------------------------------------------- #
# Final summary plot: batch size (x) vs top-1 accuracy (y), one line per method.
# Rebuilt by scanning every N*/ subdir present in RUN_DIR, so re-runs that add
# more batch sizes always refresh the full combined plot.
# ----------------------------------------------------------------------------- #
PLOT="${RUN_DIR}/batch_size_top1.png"
PLOT_CSV="${RUN_DIR}/batch_size_top1.csv"

PYTHONUNBUFFERED=1 python3 - "${RUN_DIR}" "${PLOT}" "${PLOT_CSV}" "${EVAL_SPLIT}" <<'PY'
import sys, os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir, plot_path, csv_path, eval_split = sys.argv[1:5]

rows = []
for d in sorted(glob.glob(os.path.join(run_dir, "N*"))):
    m = re.fullmatch(r"N(\d+)", os.path.basename(d))
    if not m or not os.path.isdir(d):
        continue
    n = int(m.group(1))
    csvs = sorted(glob.glob(os.path.join(d, "transductive_benchmark_summary_*.csv")))
    if not csvs:
        print(f"WARNING: no summary CSV in {d}", file=sys.stderr)
        continue
    df = pd.read_csv(csvs[-1]).dropna(subset=["top1_mean"])  # latest run for this batch size
    # best config per method (highest top-1) at this batch size
    best = df.loc[df.groupby("method")["top1_mean"].idxmax()]
    for _, r in best.iterrows():
        rows.append({"batch_size": n, "method": r["method"], "top1": r["top1_mean"]})

long = pd.DataFrame(rows).sort_values(["method", "batch_size"])
long.to_csv(csv_path, index=False)

wide = long.pivot(index="batch_size", columns="method", values="top1").sort_index()
# order legend by accuracy at the largest batch size (best on top)
order = wide.iloc[-1].sort_values(ascending=False).index.tolist()

cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(11, 7))
for i, method in enumerate(order):
    ax.plot(wide.index, wide[method], marker="o", color=cmap(i % 20), label=method)

ax.set_xlabel("test batch size (bijective N-way, N images = N queries)")
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title(f"Transductive methods vs test batch size — cross-subject LOSO ({eval_split} split, 10 subj)")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved plot: {plot_path}")
print(f"Saved CSV:  {csv_path}")
print(wide.round(2).to_string())
PY

echo "Done. All runs + summary plot under ${RUN_DIR}"
