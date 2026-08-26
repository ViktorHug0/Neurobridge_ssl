#!/bin/bash
# Cross-apply ten intra-subject EEGProject specialists to all ten test subjects.
#SBATCH --job-name=subj_spec_ens
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-subj-spec-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-subj-spec-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
python -m ensemble_experiments.evaluate_subject_specialist_ensemble \
  --device cuda:0 \
  --num-workers "${SLURM_CPUS_PER_TASK:-6}"
