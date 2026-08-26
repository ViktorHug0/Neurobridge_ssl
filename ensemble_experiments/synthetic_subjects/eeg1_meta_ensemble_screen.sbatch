#!/bin/bash
#SBATCH --job-name=eeg1_meta_ens
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/eeg1_meta_ensemble_screen/slurm-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/eeg1_meta_ensemble_screen/slurm-%j.err

set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_eeg1_meta_ensemble_screen.sh
