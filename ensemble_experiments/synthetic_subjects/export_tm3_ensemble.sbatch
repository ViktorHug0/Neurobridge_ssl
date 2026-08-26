#!/bin/bash
#SBATCH --job-name=tm3_export
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/tm3-export-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/synthetic_subjects/ensemble_screen/tm3-export-%j.err

set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/synthetic_subjects/run_export_tm3_ensemble.sh
