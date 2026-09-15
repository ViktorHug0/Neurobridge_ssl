#!/bin/bash
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m ensemble_experiments.full_sharing_v2.queue --worker 0 \
  >> results/things_eeg/full_sharing_20260914/worker0-513.log 2>&1
