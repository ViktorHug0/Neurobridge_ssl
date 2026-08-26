#!/bin/bash
# One fold of the beta=.50 fusion-aware ATM+TSConv ValCon experiment.
# Submit through dependency-linked lanes to keep a global four-GPU cap.
#SBATCH --job-name=fusion_b50
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-fusion-b50-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-fusion-b50-%j.err
set -euo pipefail

: "${SUBJECT:?submit with --export=ALL,SUBJECT=<1--6>}"

cd /nasbrain/p20fores/Neurobridge_SSL
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/fusion_wave
export ARM=fusion_valcon_atm_tsconv_b050_g000_bs512
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}
export BATCH_SIZE=512
export NUM_EPOCHS=50
export LAMBDA_DIV=0
export BETA_ENSEMBLE=0.50
export GAMMA_RESCUE=0
export RESCUE_TEMPERATURE=0.5
export RESCUE_START_EPOCH=1
export SELECTION_PROTOCOL=valcon
export VAL_CONCEPT_RATIO=0.10
export VAL_CONCEPT_SEED=20260822
export SUBJECTS="${SUBJECT}"

echo "[fusion-beta50] job=${SLURM_JOB_ID} subject=${SUBJECT} beta=${BETA_ENSEMBLE}"
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
