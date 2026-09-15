#!/bin/bash
# Five TinyTSConv and five TinyATM members x ten LOSO targets.
# Each consecutive block of ten tasks completes one target. Members alternate
# families so balanced prefixes become available during the run.
#SBATCH --job-name=tiny10_eeg
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-99%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-%A_%a.err
set -euo pipefail

member_order=(0 5 1 6 2 7 3 8 4 9)
position=$((SLURM_ARRAY_TASK_ID % 10))
export MEMBER=${member_order[$position]}
export TARGET=$((SLURM_ARRAY_TASK_ID / 10 + 1))
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/run_tiny_compute_ensemble_fold.sh
