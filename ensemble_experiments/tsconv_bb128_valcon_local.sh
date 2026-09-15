#!/bin/bash
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m ensemble_experiments.tsconv_bb128_valcon --worker 0 \
  >> results/things_eeg/tsconv_bb128_valcon_20260911/worker0-513.log 2>&1
