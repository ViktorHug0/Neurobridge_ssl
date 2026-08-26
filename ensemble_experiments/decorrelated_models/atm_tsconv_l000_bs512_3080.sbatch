#!/bin/bash
# Exact joint-training control for the heterogeneous decorrelation wave.
#SBATCH --job-name=atmts00_b512
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts00-b512-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts00-b512-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-6}
export DEVICE=cuda:0
export BATCH_SIZE=512
export LAMBDA_DIV=0.0
export ARM=decor_atm_tsconv_l000_b0_bs512
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
