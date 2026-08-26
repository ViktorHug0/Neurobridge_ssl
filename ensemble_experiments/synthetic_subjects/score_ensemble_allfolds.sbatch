#!/bin/bash
#SBATCH --job-name=vsubject_ensemble
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --nodelist=sl-mee-br-502
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/slurm-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/slurm-%j.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_score_ensemble_allfolds.sh
