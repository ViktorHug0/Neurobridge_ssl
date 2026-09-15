#!/bin/bash
#SBATCH --job-name=cov-low-514
#SBATCH --partition=Brain_GPU
#SBATCH --nodelist=sl-tp-br-514
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inductive_covariance/lowalpha_highshrink_screen/local_runner/slurm-514-%j.log

set -euo pipefail
export SUBJECTS="1 2 3"
export DEVICE=cuda:0
export NW=7
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inductive_covariance/lowalpha_highshrink_screen
export COMBOS="0.05 0.95;0.05 0.90;0.05 0.80;0.05 0.70;0.10 0.95;0.10 0.90;0.10 0.80;0.10 0.70;0.20 0.95;0.20 0.90;0.20 0.80;0.20 0.70"
bash /nasbrain/p20fores/Neurobridge_SSL/ensemble_experiments/inductive_covariance/run_grid_shard.sh
