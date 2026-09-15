#!/bin/bash
# Ten LOSO targets x eight deterministic balanced 7-of-9 source-subject bags.
#SBATCH --job-name=bag7x8_ts
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-79%4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-bag7x8-ts-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-bag7x8-ts-%A_%a.err
set -euo pipefail

export TARGET=$((SLURM_ARRAY_TASK_ID / 8 + 1))
export MEMBER=$((SLURM_ARRAY_TASK_ID % 8))
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
echo "[bag7x8-array] task=$SLURM_ARRAY_TASK_ID target=$TARGET member=$MEMBER"
bash ensemble_experiments/run_balanced_subject_bag_fold.sh
