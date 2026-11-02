#!/bin/bash
# TTA_LOG T36: REVE-d6 distilled from the 10-member fold ensemble (cv4.sh reve_d6_kdx_things builds the teacher),
# then its per-trial TTA cache, fold by fold.
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for K in ${@:-1 2 3 4}; do
  bash neurips_challenge/cv4.sh reve_d6_kdx_things $K || exit 1
done
.venv/bin/python neurips_challenge/tta_cache.py reve_d6_kdx_things
