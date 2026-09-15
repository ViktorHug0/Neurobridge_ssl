#!/bin/bash
#SBATCH --job-name=iv28-matched-control
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/iv28_control_20260909/slurm-%j.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u -m ensemble_experiments.iv28_control
