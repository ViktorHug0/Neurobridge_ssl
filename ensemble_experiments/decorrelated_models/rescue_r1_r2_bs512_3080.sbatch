#!/bin/bash
# R1/R2: soft multiple-choice rescue sweep for the ATM + TSConv pair.
#SBATCH --job-name=atmts_rescue
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts-rescue-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts-rescue-%A_%a.err
set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) export GAMMA_RESCUE=0.10; export ARM=rescue_atm_tsconv_r1_g010_b0_bs512 ;;
  1) export GAMMA_RESCUE=0.30; export ARM=rescue_atm_tsconv_r2_g030_b0_bs512 ;;
  *) echo "unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-6}
export DEVICE=cuda:0
export BATCH_SIZE=512
export LAMBDA_DIV=0
export BETA_ENSEMBLE=0
export RESCUE_TEMPERATURE=0.5
export RESCUE_START_EPOCH=1
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
