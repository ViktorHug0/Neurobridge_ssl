#!/bin/bash
# Thirty independent cohort folds: ten targets x three disjoint source triplets.
#SBATCH --job-name=cohort_ts
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-29%4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-cohort-ts-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-cohort-ts-%A_%a.err
set -euo pipefail

export TARGET=$((SLURM_ARRAY_TASK_ID / 3 + 1))
export COHORT=$((SLURM_ARRAY_TASK_ID % 3))
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
echo "[cohort-array] task=$SLURM_ARRAY_TASK_ID target=$TARGET cohort=$COHORT"
bash ensemble_experiments/run_subject_cohort_fold.sh
