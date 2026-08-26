#!/bin/bash
#SBATCH --job-name=frozen_align_dim
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-frozen-align-dim-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-frozen-align-dim-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
bash ensemble_experiments/test_selected/run_frozen_alignment_dim.sh
