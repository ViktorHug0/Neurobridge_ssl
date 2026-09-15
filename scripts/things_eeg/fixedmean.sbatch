#!/bin/bash
#SBATCH --job-name=fixmean
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=high
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subjectmix_rebuttal/%x_%j.log

# One arm, one GPU, 10 LOSO folds sequentially. Submit one job per arm.
#   sbatch --job-name=fixed_mean_k3 scripts/things_eeg/fixedmean.sbatch fixed_mean_k3
cd /nasbrain/p20fores/Neurobridge_SSL
srun ./run_fixedmean_paperbase.sh "$1" 1 2 3 4 5 6 7 8 9 10
