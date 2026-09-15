#!/bin/bash
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ ! -f results/things_eeg/tsconv_bb128_valcon_20260911/summary.json ]; do sleep 30; done
python -u -m ensemble_experiments.full_sharing --worker 0 \
  >> results/things_eeg/full_sharing_20260911/worker-513.log 2>&1
