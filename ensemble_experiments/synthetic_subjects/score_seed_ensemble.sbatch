#!/bin/bash
#SBATCH --job-name=seed_ensemble
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/seed-ensemble-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/seed-ensemble-%j.err

set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_seed_ensemble_allfolds.sh
