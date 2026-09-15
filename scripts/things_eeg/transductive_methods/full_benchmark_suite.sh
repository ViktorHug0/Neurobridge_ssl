#!/usr/bin/env bash
#
# Full transductive benchmark suite: runs EVERY method once, each with its tuned best
# config (--use_best_configs), on the same encoded (query, image) pair per subject.
# Combines transductive_lambda_sweep.sh + transclip_sweep.sh, minus the TransCLIP ablations.
#
# - SATTC always uses SAW(0.94)+CSLS + the tuned refinement (tau=0.1, steps=14, pow=1.1, iters=14).
# - Each other method uses the preprocessing/hyperparams that were best in the sweeps
#   (see BEST_CONFIGS in transductive_benchmark.py): TransCLIP / hungarian / csls / plain_cosine
#   run on raw features; the EM / k-means / sinkhorn heads run on SAW+CSLS.
# - Records mean per-method inference duration; writes a comparison barplot at the end.
#
# Eval-set options (env vars):
#   EVAL_SPLIT=test|train   default test. 'train' = left-out subject's train recordings,
#                           EEG averaged over the 4 reps/image.
#   SUBSAMPLE=N             bijective N-way task (sets n_images=n_queries=N). e.g. SUBSAMPLE=100
#   N_IMAGES=K / N_QUERIES=N   choose prototypes K and queries N (N<=K) independently.
#
# Examples:
#   bash full_benchmark_suite.sh                          # full 200-way test, 10 subjects
#   SUBSAMPLE=100 bash full_benchmark_suite.sh            # 100-way test
#   EVAL_SPLIT=train N_IMAGES=200 N_QUERIES=200 bash full_benchmark_suite.sh
#   SOURCE_RUN_DIR=<param dir> bash full_benchmark_suite.sh

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
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-results/full_benchmark_suite/${STAMP}_${EVAL_SPLIT}}"
LOG_FILE="${OUTPUT_DIR}/suite.log"

mkdir -p "${OUTPUT_DIR}"

# Eval-set subsampling flags.
EVAL_FLAGS=(--eval_split "${EVAL_SPLIT}")
[ -n "${SUBSAMPLE:-}" ] && EVAL_FLAGS+=(--subsample "${SUBSAMPLE}")
[ -n "${N_IMAGES:-}" ]  && EVAL_FLAGS+=(--n_images "${N_IMAGES}")
[ -n "${N_QUERIES:-}" ] && EVAL_FLAGS+=(--n_queries "${N_QUERIES}")

echo "Source run dir : ${SOURCE_RUN_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Eval split     : ${EVAL_SPLIT}"
echo "Device         : ${DEVICE}"

PYTHONUNBUFFERED=1 python3 scripts/things_eeg/transductive_methods/transductive_benchmark.py \
    --source_run_dir "${SOURCE_RUN_DIR}" \
    --held_out_subjects 1 2 3 4 5 6 7 8 9 10 \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --use_best_configs --plot \
    --methods full_sattc sattc_hungarian hungarian_sattc transclip plain_cosine csls sinkhorn hungarian \
              kl_kmeans soft_kmeans hard_kmeans \
              em_dirichlet hard_em_dirichlet em_gaussian em_gaussian_cov \
    --sattc_csls_k 1 --sattc_sinkhorn_tau 0.1 --sattc_sinkhorn_iters 14 \
    --sattc_soft_procrustes_steps 14 --sattc_soft_procrustes_power 1.1 \
    "${EVAL_FLAGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "Done. Summary + per-subject CSVs + barplot in ${OUTPUT_DIR}"
