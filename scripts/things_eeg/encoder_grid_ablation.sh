#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Paths to the CSV files
NOMIXUP_CSV="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/eeg_encoder_architecture_sweep_nomixup_20260429-221100/sweep_summary.csv"
MIXUP_CSV="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/eeg_encoder_architecture_sweep_mixup_20260429-221100/sweep_summary.csv"

# Find the latest TTA sweep summary
TTA_DIR=$(ls -d /nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/tta_sweep_* | sort | tail -n 1)
TTA_CSV="${TTA_DIR}/tta_sweep_summary.csv"

# Output folder for plots
OUTPUT_DIR="${REPO_ROOT}/results/things_eeg/inter-subjects/encoder_grid_ablation_plots"

IGNORE_CONFORMER_FLAG=()
for arg in "$@"; do
    if [[ "$arg" == "--ignore-conformer" ]]; then
        IGNORE_CONFORMER_FLAG=(--ignore_conformer)
    fi
done

echo "Generating encoder grid ablation plots..."
echo "No SubjectMix sweep CSV: $NOMIXUP_CSV"
echo "SubjectMix sweep CSV: $MIXUP_CSV"
echo "TTA CSV: $TTA_CSV"
echo "Output Directory: $OUTPUT_DIR"
if ((${#IGNORE_CONFORMER_FLAG[@]})); then
    echo "Ignoring EEGConformer rows"
fi

python3 "${SCRIPT_DIR}/encoder_grid_ablation.py" \
    --nomixup_csv "$NOMIXUP_CSV" \
    --mixup_csv "$MIXUP_CSV" \
    --tta_csv "$TTA_CSV" \
    --output_dir "$OUTPUT_DIR" \
    "${IGNORE_CONFORMER_FLAG[@]}"

echo "Done! Plots saved in $OUTPUT_DIR"
