#!/usr/bin/env bash
# Minimal test of the hypothesis: does SATTC need MORE iterations at larger N?
# Runs full_sattc only (train split, 10 LOSO subjects) at N in {500,1000},
# scaling soft-Procrustes steps and Sinkhorn iters together (1x/2x/4x baseline).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

SRC="results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="results/sattc_iter_scaling/${STAMP}"
mkdir -p "${RUN_DIR}"
echo "Run dir: ${RUN_DIR}"

for N in 500 1000; do
  for IT in 14 28 56; do      # baseline 14, then 2x, 4x (steps == sinkhorn_iters)
    OUT="${RUN_DIR}/N${N}_it${IT}"
    echo "=== N=${N} steps=iters=${IT} -> ${OUT} ==="
    PYTHONUNBUFFERED=1 python3 scripts/things_eeg/transductive_methods/transductive_benchmark.py \
      --source_run_dir "${SRC}" \
      --held_out_subjects 1 2 3 4 5 6 7 8 9 10 \
      --output_dir "${OUT}" --device cuda:0 \
      --use_best_configs --methods full_sattc \
      --sattc_csls_k 1 --sattc_sinkhorn_tau 0.1 \
      --sattc_sinkhorn_iters "${IT}" --sattc_soft_procrustes_steps "${IT}" \
      --sattc_soft_procrustes_power 1.1 \
      --eval_split train --subsample "${N}"
  done
done

echo "=== SUMMARY (full_sattc top1 mean over 10 subj) ==="
python3 - "${RUN_DIR}" <<'PY'
import sys, os, glob, re, pandas as pd
run=sys.argv[1]; rows=[]
for d in sorted(glob.glob(os.path.join(run,"N*_it*"))):
    m=re.fullmatch(r"N(\d+)_it(\d+)", os.path.basename(d))
    if not m: continue
    c=sorted(glob.glob(os.path.join(d,"transductive_benchmark_summary_*.csv")))
    if not c: continue
    df=pd.read_csv(c[-1]); r=df[df.method=="full_sattc"].iloc[0]
    rows.append({"N":int(m.group(1)),"iters":int(m.group(2)),
                 "top1":r.top1_mean,"top5":r.top5_mean,"dur_s":round(r.duration_mean,3)})
t=pd.DataFrame(rows).pivot_table(index="N",columns="iters",values="top1")
print("top1 by N x iters:\n", t.round(2).to_string())
print("\nfull table:\n", pd.DataFrame(rows).sort_values(["N","iters"]).to_string(index=False))
PY
