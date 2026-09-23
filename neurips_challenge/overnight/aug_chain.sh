#!/bin/bash
# Neurobridge image augmentation on T2 + trunk 1024, constant lr: verify -> extract -> fuse -> train.
source "$(dirname "$0")/common.sh"
CPY=/nasbrain/p20fores/Neurips_challenge/.venv/bin/python
AUGDIR=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianBlur-GaussianNoise-LowResolution-Mosaic
set -e
echo "=== $(date '+%F %T') verify"; $CPY neurips_challenge/build_aug_targets.py --verify 500
echo "=== $(date '+%F %T') extract"; $CPY neurips_challenge/build_aug_targets.py
set +e
step env AVERAGE= bash $R TSConv_parameterizable t2_trunk1024_const_aug \
    --tsconv_pool_kernel 25 --tsconv_pool_stride 2 --num_epochs 100 --lr_scheduler none --grad_clip 0 \
    --tsconv_backbone_dim 1024 --image_aug --aug_image_feature_dirs "$AUGDIR"
