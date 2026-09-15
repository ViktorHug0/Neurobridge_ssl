#!/bin/bash
# One forward pass over the THINGS image set, hooking InternViT layers 35/39/44.
# Layer 33 is hooked too as a free correctness check: it costs nothing extra in
# the same pass and must reproduce the existing layer-33 features.
#SBATCH --job-name=iv_layers
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-iv-layers-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-iv-layers-%j.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
# compute nodes cannot write $HOME/.triton over NFS; bitsandbytes needs a cache
export TRITON_CACHE_DIR="/tmp/$USER-triton-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR"
trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

FEAT="$REPO_ROOT/data/things_eeg/image_feature"
STAGE="$FEAT/_stage_iv_layer{layer}"

python -u extract_internvit_layers.py \
  --layers 33 35 39 44 \
  --output-template "$STAGE" \
  --batch-size 8 \
  --device cuda:0

# Layer 33 must match the existing extraction, otherwise the new layers are not
# comparable with the arms already trained on this backbone.
python - <<'PY'
import numpy as np, sys
from pathlib import Path
new = Path("data/things_eeg/image_feature/_stage_iv_layer33")
old = Path("data/things_eeg/image_feature/InternViT-6B_layer33_mean_8bit")
worst = 0.0
for name in ("image_train.npy", "image_test.npy"):
    a = np.load(new / name).astype(np.float32)
    b = np.load(old / name).astype(np.float32)
    assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)
    print(f"{name}: min cosine {cos.min():.6f}  mean cosine {cos.mean():.6f}  max rel err {rel:.2e}")
    worst = max(worst, 1.0 - float(cos.min()))
if worst > 1e-3:
    sys.exit(f"layer-33 reproduction failed (1-cos = {worst:.2e}); new layers NOT promoted")
print("layer-33 check passed")
PY

for L in 35 39 44; do
  mv "$FEAT/_stage_iv_layer$L" "$FEAT/InternViT-6B_layer${L}_mean_8bit"
  echo "promoted layer $L -> $FEAT/InternViT-6B_layer${L}_mean_8bit"
done
rm -rf "$FEAT/_stage_iv_layer33"
ls -d "$FEAT"/InternViT-6B_layer*_mean_8bit
