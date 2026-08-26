#!/bin/bash
#SBATCH --job-name=rep_allfold
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/repetition_ensemble_allfolds/slurm-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/repetition_ensemble_allfolds/slurm-%j.err

set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_repetition_ensemble_allfolds.sh
