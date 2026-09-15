#!/bin/bash
# Phase A on-GPU check: gate, then a short real-data run to measure the paired
# epoch time the screening budget depends on.
#SBATCH --job-name=tinyshare-smoke
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tiny_sharing/smoke-%j.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

python -m ensemble_experiments.tiny_sharing.gate

python -m ensemble_experiments.tiny_sharing.train_pair \
  --slots 1 --configs C1 C0 C9 --epochs 2 \
  --result-root /nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tiny_sharing/smoke
