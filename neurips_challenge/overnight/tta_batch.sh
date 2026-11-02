#!/bin/bash
# TTA_LOG batch runner: each argument line = one tta_lab.py invocation (quoted), run in order on this GPU.
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for line in "$@"; do
  echo "=== $line"
  .venv/bin/python neurips_challenge/tta_lab.py $line 2>&1 | grep "RESULT\|purity\|Error"
done
