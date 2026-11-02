#!/bin/bash
# TTA_LOG T19: batch-size robustness. args: <batch sizes...>
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=base+fine+calpha+cbeta+cgamma+pgamma
for B in "$@"; do
  Ks=$(( B / 16 )); [ $Ks -lt 1 ] && Ks=1
  echo "=== B $B (scaled K $Ks)"
  TTA_BS=$B .venv/bin/python neurips_challenge/tta_lab.py ref wknn:4:0.7 wknn:$Ks:0.7 wknng:4:0.7:$R wknng:$Ks:0.7:$R 2>&1 | grep "RESULT\|Error"
done
