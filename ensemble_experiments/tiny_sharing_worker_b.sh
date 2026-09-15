#!/bin/bash
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m ensemble_experiments.tiny_sharing_two_gpu --worker B \
  >> results/things_eeg/tiny_sharing/worker-B-513.log 2>&1
