#!/bin/bash
#SBATCH --job-name=compact-valcon
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=1-2%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/compact_valcon_20260908/logs/slurm-%A_%a.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u -m ensemble_experiments.compact_valcon.run_queue --worker "$SLURM_ARRAY_TASK_ID"
