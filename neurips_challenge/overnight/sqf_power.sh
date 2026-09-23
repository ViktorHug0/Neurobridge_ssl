#!/bin/bash
# Squeezeformer + fused band-power branch, else identical to sqf_trunk512_oc50 (0.3727).
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqf_trunk512_oc50_power --tsconv_pool_kernel 25 --tsconv_pool_stride 2 \
    --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch
