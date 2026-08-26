#!/bin/bash
# One-epoch gate for the new concept-validation and dual-checkpoint path.
#SBATCH --job-name=rescue_vc_smoke
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-rescue-vc-smoke-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-rescue-vc-smoke-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export ARM=rescue_valcon_smoke_e1
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/smoke
export SUBJECTS=1
export NW=${SLURM_CPUS_PER_TASK:-6}
export DEVICE=cuda:0
export BATCH_SIZE=512
export NUM_EPOCHS=1
export LAMBDA_DIV=0
export BETA_ENSEMBLE=0
export GAMMA_RESCUE=0.10
export RESCUE_TEMPERATURE=0.5
export RESCUE_START_EPOCH=1
export SELECTION_PROTOCOL=valcon
export VAL_CONCEPT_RATIO=0.10
export VAL_CONCEPT_SEED=20260822
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
