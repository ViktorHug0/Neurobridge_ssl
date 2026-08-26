#!/bin/bash
# Prepared only: fit one router on all ten subjects and apply that exact state
# to all ten. Subject identity is not an input and cannot change the rule.
#SBATCH --job-name=ens50_pool_router
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-pooled-router-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-pooled-router-%j.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

python ensemble_experiments/test_selected/learned_pooled_router.py \
  --config ensemble_experiments/test_selected/pooled_router_config.json \
  --device cpu \
  --output-dir results/things_eeg/ensemble50_testselected/pooled_router
