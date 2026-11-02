#!/bin/bash
# Squeezeformer (0.3727 recipe) and FBTN (0.3769) with an EMA of the weights.
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqf_trunk512_oc50_ema --tsconv_pool_kernel 25 --tsconv_pool_stride 2 \
    --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --ema_decay 0.9995 --ema_warmup
step env AVERAGE= bash $R FilterBankTangentNet fbtn_ema --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --arch_kwargs '{}' \
    --ema_decay 0.9995 --ema_warmup
