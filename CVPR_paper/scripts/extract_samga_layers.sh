#!/bin/bash
# InternViT-6B layers 20/24/32/36 for SAMGA's multi-layer router (SCORE original stack). Layer 28 is hooked in the same
# pass as a correctness check against the existing layer-28 features. bitsandbytes lives only in the sibling NeuroBridge
# venv (the one `source .venv/bin/activate` resolves to, as in scripts/things_eeg/extract_internvit_layers.sbatch).
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
export TRITON_CACHE_DIR=/tmp/$USER-triton-${SLURM_JOB_ID:-x}; mkdir -p $TRITON_CACHE_DIR
PY=/nasbrain/p20fores/NeuroBridge/.venv/bin/python
FEAT=data/things_eeg/image_feature
$PY -u extract_internvit_layers.py --layers 20 24 28 32 36 --output-template "$FEAT/_stage_iv_layer{layer}" --batch-size 8
$PY - <<'PY'
import numpy as np
a = np.load('data/things_eeg/image_feature/_stage_iv_layer28/image_test.npy')
b = np.load('data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit/image_test.npy')
c = (a.reshape(len(a), -1) * b.reshape(len(b), -1)).sum(1) / np.linalg.norm(a.reshape(len(a), -1), axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1)
c = c.astype(np.float64); print('layer28 check: min cosine', c.min()); assert c.min() > 0.998  # float16 features: 8-bit re-extraction gives 0.99902
PY
for L in 20 24 32 36; do mv $FEAT/_stage_iv_layer$L $FEAT/InternViT-6B_layer${L}_mean_8bit; done
rm -rf $FEAT/_stage_iv_layer28
