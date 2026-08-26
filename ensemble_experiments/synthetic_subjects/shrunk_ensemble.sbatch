#!/bin/bash
#SBATCH --job-name=shrunk_ens
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/shrunk_ensemble/slurm-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/shrunk_ensemble/slurm-%A_%a.err
set -euo pipefail
ARMS=(eva_s3300 vith_s3300 eva_s3301 eva_s3302)
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_shrunk_ensemble.sh "${ARMS[$SLURM_ARRAY_TASK_ID]}"
