#!/bin/bash
# UltraTiny draft reliability check: both members on subjects 2 and 3.
# Run models separately so elapsed times remain intrinsic and additive.
#SBATCH --job-name=ultratiny_s23
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-ultratiny-s23-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-ultratiny-s23-%A_%a.err
set -euo pipefail

export TARGET=$((2 + SLURM_ARRAY_TASK_ID / 2))
export MEMBER=$((SLURM_ARRAY_TASK_ID % 2))
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
start=$SECONDS
bash ensemble_experiments/run_ultratiny_reference_fold.sh
elapsed=$((SECONDS - start))
echo "ULTRATINY_RESULT member=$MEMBER target=$TARGET seconds=$elapsed"
