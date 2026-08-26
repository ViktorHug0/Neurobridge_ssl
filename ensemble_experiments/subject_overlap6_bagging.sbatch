#!/bin/bash
# Thirty folds: ten targets x three balanced, overlapping six-source TSConv bags.
# Submit with a dependency on the triplet wave to preserve the global four-GPU cap.
#SBATCH --job-name=overlap6_ts
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-29%4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-overlap6-ts-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-overlap6-ts-%A_%a.err
set -euo pipefail

export TARGET=$((SLURM_ARRAY_TASK_ID / 3 + 1))
export COHORT=$((SLURM_ARRAY_TASK_ID % 3))
export GROUP_MODE=overlap6
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_cohort_bagging/testselected_overlap6
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
echo "[overlap6-array] task=$SLURM_ARRAY_TASK_ID target=$TARGET cohort=$COHORT"
bash ensemble_experiments/run_subject_cohort_fold.sh
