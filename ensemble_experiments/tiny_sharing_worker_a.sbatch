#!/bin/bash
#SBATCH --job-name=tinyshare-A
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --nodelist=sl-tp-br-515
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tiny_sharing/worker-A-%j.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m ensemble_experiments.tiny_sharing_two_gpu --worker A
