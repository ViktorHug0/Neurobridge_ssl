#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

SOURCE_DIR="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300"
RUN_TAG="progressive_regime_study_20260506-163658" # "progressive_regime_study_$(date +'%Y%m%d-%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/results/things_eeg/inter-subjects/${RUN_TAG}"

echo "Starting progressive TTA experiment (Two Regimes)..."
echo "Source: ${SOURCE_DIR}"
echo "Output: ${OUTPUT_DIR}"

# python3 "${SCRIPT_DIR}/progressive_sattc_candidate_sweep.py" \
#     --source_run_dir "$SOURCE_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --min_samples 5 \
#     --max_samples 200 \
#     --sample_step 5 \
#     --sattc_saw_shrink 0.94 \
#     --sattc_csls_k 3 \
#     --sattc_sinkhorn_tau 0.1 \
#     --sattc_sinkhorn_iters 12 \
#     --sattc_soft_procrustes_steps 8 \
#     --sattc_soft_procrustes_power 1.2 \
#     --held_out_subjects 1 2 3 4 5 6 7 8 9 10 \
#     --seeds 3300 3301 3302

echo "Generating comparison plots..."
python3 "${SCRIPT_DIR}/plot_progressive_sattc_results.py" \
    --csv_path "${OUTPUT_DIR}/progressive_sattc_results.csv"

echo "Experiment complete. Results saved in ${OUTPUT_DIR}"
