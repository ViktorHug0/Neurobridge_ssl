#!/bin/bash
# First UltraTiny draft: intrinsic end-to-end timing and accuracy on subject 1.
# Models run separately (no MPS) so their elapsed times sum to the pair cost.
#SBATCH --job-name=ultratiny_s1
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-ultratiny-s1-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-ultratiny-s1-%A_%a.err
set -euo pipefail

export TARGET=1
export MEMBER=$SLURM_ARRAY_TASK_ID
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
start=$SECONDS
bash ensemble_experiments/run_ultratiny_reference_fold.sh
elapsed=$((SECONDS - start))
echo "ULTRATINY_RESULT member=$MEMBER target=$TARGET seconds=$elapsed"
