#!/bin/bash
# Prepared only: build the leakage-checked manifest and run the fixed router.
# This is CPU-only and must depend on successful completion of every dump task.
#SBATCH --job-name=ens50_router_fit
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-fit-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-fit-%j.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

MANIFEST=results/things_eeg/ensemble50_testselected/learned_router/manifest.json
python ensemble_experiments/test_selected/build_router_manifest.py --output "$MANIFEST"
python ensemble_experiments/test_selected/learned_global_router.py \
  --manifest "$MANIFEST" --device cpu \
  --output-dir results/things_eeg/ensemble50_testselected/learned_router/router_runs
