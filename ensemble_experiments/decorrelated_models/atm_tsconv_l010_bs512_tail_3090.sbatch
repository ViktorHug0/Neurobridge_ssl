#!/bin/bash
# Finish subjects 9-10 of the lambda=.10, batch-512 arm in parallel with
# the original 3080 worker, which is expected to finish subjects 7-8.
#SBATCH --job-name=atmts10_tail
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts10-tail-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts10-tail-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-8}
export DEVICE=cuda:0
export BATCH_SIZE=512
export ARM=decor_atm_tsconv_l010_b0_bs512
export SUBJECTS="9 10"
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
