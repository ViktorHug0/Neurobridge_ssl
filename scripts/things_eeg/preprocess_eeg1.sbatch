#!/bin/bash
#SBATCH --job-name=prep_eeg1
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/prep_eeg1_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/prep_eeg1_%j.err

# THINGS-EEG1 (ds003825) -> Things-EEG1/Preprocessed_data_250Hz, EEG2 layout.
# CPU only. Resumable: skips any subject whose preprocessed_eeg_training.npy exists.
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
echo "host=$(hostname) start=$(date)"
python preprocess_eeg1.py --mvnn --subjects "${SUBJECTS:-}" ${EXTRA_ARGS:-}
echo "DONE $(date)"
