#!/bin/bash
# Phase 0: extract the 4 Neurobridge image augmentations as final ViT-H/14 features, then
# average them into one aug feature dir (matches Neurobridge CPA: average of augmented views).
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

ROOT=data/things_eeg/image_feature
AUGS=(GaussianBlur GaussianNoise LowResolution Mosaic)

for A in "${AUGS[@]}"; do
  OUT="$ROOT/ViT-H-14_aug_$A"
  if [ ! -f "$OUT/train.npy" ]; then
    echo "=== extracting ViT-H aug: $A ==="
    python extract_feature.py --model_type open_clip --backbone ViT-H-14 \
      --pretrained laion2b_s32b_b79k --feature_source final \
      --aug_type "$A" --output_dir "$OUT"
  fi
done

echo "=== averaging 4 augmented views -> ViT-H-14_final_augavg ==="
python - <<'PY'
import numpy as np, os
root="data/things_eeg/image_feature"
augs=["GaussianBlur","GaussianNoise","LowResolution","Mosaic"]
out=os.path.join(root,"ViT-H-14_final_augavg"); os.makedirs(out,exist_ok=True)
for split in ["train","test"]:
    stk=[np.load(os.path.join(root,f"ViT-H-14_aug_{a}",f"{split}.npy")).astype(np.float32) for a in augs]
    avg=np.mean(np.stack(stk,0),0)          # (1,objects,images,D) mean over the 4 augs
    np.save(os.path.join(out,f"{split}.npy"), avg.astype(np.float16))
    print(split, avg.shape, "norm~", np.linalg.norm(avg.reshape(-1,avg.shape[-1]),axis=1).mean())
PY
echo "PHASE0 DONE"
