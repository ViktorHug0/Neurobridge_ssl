#!/bin/bash
# Step 2: single-trial augmentation for the scratch models (val split kept to measure each effect).
source "$(dirname "$0")/common.sh"
SQF="--tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch"
FBTN="--grad_clip 0 --early_stop_patience 0 --num_epochs 50 --arch_kwargs {}"
case "$1" in
  0) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqfp_xmix05 $SQF --xsubj_mix_prob 0.5 ;;
  1) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqfp_xmix10 $SQF --xsubj_mix_prob 1.0 ;;
  2) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqfp_amp_tmask $SQF --amp_scale_std 0.1 --time_mask_len 12 ;;
  3) step env AVERAGE= bash $R FilterBankTangentNet fbtn_xmix05 $FBTN --xsubj_mix_prob 0.5 ;;
esac
