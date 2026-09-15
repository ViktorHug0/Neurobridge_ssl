#!/usr/bin/env bash
#
# Lambda (class-balance) sweep for the EM transductive heads across all 10 LOSO
# subjects on the 200-way THINGS-EEG test set, alongside the fixed baselines
# (incumbent full_sattc, full_sattc+Hungarian, plain cosine, CSLS, Sinkhorn,
# Hungarian, k-means / KL-means).
#
# Evaluation is CROSS-SUBJECT (LOSO): each checkpoint is trained on the 9 other
# subjects and tested on the held-out one.
#
# Every method shares the same front-end: SAW whitening (shrink=0.94) + CSLS (k=1),
# the hand-picked values. full_sattc / sattc_hungarian additionally use the hand-picked
# refinement config (tau=0.1, steps=14, power=1.1, sinkhorn_iters=14).
#
# Sweep: lambda in {0.05, 0.1, 0.25} + {0.5, 1.0, ..., 10.0}  (23 values)
#        temperature T in {50, 60, 70, 80, 90, 100, 110, 120}  (8 values).
# EM heads sweep BOTH lambda and T (184 configs each); k-means heads sweep T only.

# Launch:  bash scripts/things_eeg/transductive_lambda_sweep.sh
# Override the source run dir / seed:  SOURCE_RUN_DIR=<param dir with *-sub-XX subdirs> bash <this>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# ----------------------------------------------------------------------------- #
# Configuration (override via environment).
# ----------------------------------------------------------------------------- #
# LOSO checkpoints: a "param_*" dir that DIRECTLY contains one *-sub-XX subdir per
# subject (train on the 9 other subjects, test on the held-out one).
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"
DEVICE="${DEVICE:-cuda:0}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-results/transductive_lambda_sweep/${STAMP}}"
LOG_FILE="${OUTPUT_DIR}/sweep.log"

mkdir -p "${OUTPUT_DIR}"

echo "Source run dir : ${SOURCE_RUN_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Log file       : ${LOG_FILE}"
echo "Device         : ${DEVICE}"

PYTHONUNBUFFERED=1 python3 scripts/things_eeg/transductive_methods/transductive_benchmark.py \
    --source_run_dir "${SOURCE_RUN_DIR}" \
    --held_out_subjects 1 2 3 4 5 6 7 8 9 10 \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --methods full_sattc sattc_hungarian plain_cosine csls sinkhorn hungarian \
              kl_kmeans soft_kmeans hard_kmeans \
              em_dirichlet hard_em_dirichlet em_gaussian em_gaussian_cov \
    --saw_all \
    --sattc_saw_shrink 0.94 \
    --sattc_csls_k 1 \
    --sattc_sinkhorn_tau 0.1 \
    --sattc_sinkhorn_iters 14 \
    --sattc_soft_procrustes_steps 14 \
    --sattc_soft_procrustes_power 1.1 \
    --em_lambdas 0.05 0.1 0.25 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 \
    --feature_Ts 50 60 70 80 90 100 110 120 \
    --em_iter 20 --em_iter_mm 50 \
    2>&1 | tee "${LOG_FILE}"

echo "Done. Summary + per-subject CSVs in ${OUTPUT_DIR}"
