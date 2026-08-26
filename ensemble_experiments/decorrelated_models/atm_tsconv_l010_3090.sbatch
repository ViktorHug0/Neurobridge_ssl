#!/bin/bash
#SBATCH --job-name=decor_atmts10
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=14:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-decor-atmts10-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-decor-atmts10-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-8}
export DEVICE=cuda:0
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
