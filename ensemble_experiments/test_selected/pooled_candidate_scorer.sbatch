#!/bin/bash
# Two fixed global candidate scorers, one per GPU. Both use all ten subjects and
# one byte-identical scorer per variant; subject identity is never an input.
#SBATCH --job-name=ens50_cand
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-candidate-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-candidate-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

VARIANTS=(linear mlp)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
python ensemble_experiments/test_selected/learned_pooled_candidate_scorer.py \
  --config ensemble_experiments/test_selected/pooled_candidate_config.json \
  --variant "$VARIANT" --device cuda:0 \
  --output-root results/things_eeg/ensemble50_testselected/pooled_candidate_scorer
