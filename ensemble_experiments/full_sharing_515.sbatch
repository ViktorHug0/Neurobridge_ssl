#!/bin/bash
#SBATCH --job-name=fullshare
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --nodelist=sl-tp-br-515
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/full_sharing_20260911/worker-515-%j.log
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ ! -f results/things_eeg/tsconv_bb128_valcon_20260911/summary.json ]; do sleep 30; done
python -u -m ensemble_experiments.full_sharing --worker 1
