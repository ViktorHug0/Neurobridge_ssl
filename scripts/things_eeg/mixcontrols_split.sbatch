#!/bin/bash
#SBATCH --job-name=mixctrl
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subjectmix_rebuttal/%x_%j.log

# One arm split across two GPUs: submit twice with disjoint subject lists.
#   sbatch --job-name=mixco_a scripts/things_eeg/mixcontrols_split.sbatch mixco 1 2 3 4 5
#   sbatch --job-name=mixco_b scripts/things_eeg/mixcontrols_split.sbatch mixco 6 7 8 9 10
# The runner skips any fold that already has a result.csv, so the halves cannot collide.
cd /nasbrain/p20fores/Neurobridge_SSL
srun ./run_mixcontrols_paperbase.sh "$@"
