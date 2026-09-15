#!/bin/bash
#SBATCH --job-name=fullshare-third
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513,sl-tp-br-514
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/full_sharing_20260914/worker2-%j.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m ensemble_experiments.full_sharing_third
