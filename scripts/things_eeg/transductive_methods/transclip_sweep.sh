#!/usr/bin/env bash
#
# TransCLIP (group B) sweep on the 200-way THINGS-EEG LOSO test set, all 10 subjects.
# Separate from transductive_lambda_sweep.sh (which sweeps the EM / k-means / SATTC heads).
#
# Sweeps TransCLIP's own knobs:  text-KL weight lambda in {0, 0.5, 1, 2, 4}
#                                k-NN neighbors in {3, 5, 10}
# over the full TransCLIP-ZS head and its ablations, in two preprocessing arms:
#   - raw features                (this script)
#   - SAW-whitened features       (set SAW=1)   <- matches the front-end full_sattc enjoys
#
# Baselines full_sattc (tuned) and plain_cosine are included for reference bars.
# TransCLIP runs on GPU. Variants that don't use lambda/n_neighbors still re-run per grid
# point (cheap), so filter the CSV to the relevant knobs per variant when plotting.
#
# Launch:        bash scripts/things_eeg/transclip_sweep.sh
# With SAW:      SAW=1 bash scripts/things_eeg/transclip_sweep.sh
# Override dir:  SOURCE_RUN_DIR=<param dir with *-sub-XX> bash scripts/things_eeg/transclip_sweep.sh

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
SAW="${SAW:-0}"                       # SAW=1 to whiten features before TransCLIP
STAMP="$(date +%Y%m%d-%H%M%S)"
ARM=$([ "${SAW}" = "1" ] && echo "saw" || echo "raw")
OUTPUT_DIR="${OUTPUT_DIR:-results/transclip_sweep/${STAMP}_${ARM}}"
LOG_FILE="${OUTPUT_DIR}/sweep.log"

mkdir -p "${OUTPUT_DIR}"

SAW_FLAGS=()
if [ "${SAW}" = "1" ]; then
    SAW_FLAGS=(--saw_all --sattc_saw_shrink 0.94)
fi

echo "Source run dir : ${SOURCE_RUN_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Preprocessing  : ${ARM}"
echo "Device         : ${DEVICE}"

PYTHONUNBUFFERED=1 python3 scripts/things_eeg/transductive_methods/transductive_benchmark.py \
    --source_run_dir "${SOURCE_RUN_DIR}" \
    --held_out_subjects 1 2 3 4 5 6 7 8 9 10 \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --methods transclip transclip_no_kl transclip_no_lap \
              transclip_mu_only transclip_sigma_only transclip_anchor \
              full_sattc plain_cosine \
    "${SAW_FLAGS[@]}" \
    --sattc_csls_k 1 --sattc_sinkhorn_tau 0.1 --sattc_sinkhorn_iters 14 \
    --sattc_soft_procrustes_steps 14 --sattc_soft_procrustes_power 1.1 \
    --transclip_lambdas 0 0.5 1 2 4 \
    --transclip_neighbors_list 3 5 10 \
    --transclip_max_iter 10 \
    2>&1 | tee "${LOG_FILE}"

echo "Done. Summary + per-subject CSVs in ${OUTPUT_DIR}"
