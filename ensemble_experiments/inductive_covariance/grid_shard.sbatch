#!/bin/bash
#SBATCH --job-name=cov-grid-514
#SBATCH --partition=Brain_GPU
#SBATCH --nodelist=sl-tp-br-514
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inductive_covariance/single_tsconv_screen/local_runner/slurm-514-%j.log

set -euo pipefail
export SUBJECTS="9 10"
export DEVICE=cuda:0
export NW=6
bash /nasbrain/p20fores/Neurobridge_SSL/ensemble_experiments/inductive_covariance/run_grid_shard.sh
