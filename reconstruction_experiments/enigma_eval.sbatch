#!/bin/bash
#SBATCH --job-name=enigma_eval
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_eval_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_eval_%j.err

# Stage 3 only: train_logs/ and output/ from job 48496 are already on disk. evaluate_recons.py
# imports openai CLIP, which is in their environment.yml but missing from requirements.txt.
set -e
cd /nasbrain/p20fores/ENIGMA
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
export WANDB_MODE=offline
/nasbrain/p20fores/ENIGMA/.venv-enigma/bin/python evaluate_recons.py \
    --model_name ENIGMA_repro_sub01 --subj_ids 1 --config_name things_eeg2_local
