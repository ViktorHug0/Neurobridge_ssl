#!/bin/bash
# Step 1: port the foundation-model recipe pieces to the scratch models.
#   cd   = 10% channel dropout (val split kept, so the effect is isolated)
#   full = all training concepts, full 50-epoch anneal, last epoch kept
source "$(dirname "$0")/common.sh"
SQF="OrthoFastTSConvSqueezeformer --tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch"
FBTN="FilterBankTangentNet --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --arch_kwargs {}"
CD="--eeg_aug --eeg_aug_type channel_dropout"
FULL="--val_concept_ratio 0 --select_best_on fixed"
case "$1" in
  0) step env AVERAGE= bash $R ${SQF%% *} sqfp_cd ${SQF#* } $CD ;;
  1) step env AVERAGE= bash $R ${FBTN%% *} fbtn_cd ${FBTN#* } $CD ;;
  2) step env AVERAGE= bash $R ${SQF%% *} sqfp_full ${SQF#* } $FULL ;;
  3) step env AVERAGE= bash $R ${FBTN%% *} fbtn_full ${FBTN#* } $FULL ;;
  4) step env AVERAGE= bash $R ${SQF%% *} sqfp_full_cd ${SQF#* } $FULL $CD ;;
  5) step env AVERAGE= bash $R ${FBTN%% *} fbtn_full_cd ${FBTN#* } $FULL $CD ;;
esac
