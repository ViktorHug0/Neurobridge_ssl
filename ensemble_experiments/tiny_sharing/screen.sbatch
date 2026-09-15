#!/bin/bash
# Phase B: ten configurations x three inner screening folds on one RTX 3080.
# One array task per slot, serialized (%1) so only one GPU job runs at a time.
#SBATCH --job-name=tinyshare-screen
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=1-3%1
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tiny_sharing/screen-%A_%a.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# C0 and C1 first: the bridge gate is read off slot 1 before the grid fills.
python -m ensemble_experiments.tiny_sharing.train_pair \
  --slots "$SLURM_ARRAY_TASK_ID" \
  --configs C0 C1 C2 C3 C4 C5 C6 C7 C8 C9
