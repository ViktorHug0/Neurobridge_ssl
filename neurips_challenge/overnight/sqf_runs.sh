#!/bin/bash
# Squeezeformer follow-ups to sqf_trunk512_oc50 (0.3727). usage: sqf_runs.sh e100|wide|trunk1024
source "$(dirname "$0")/common.sh"
SQF="--tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0"
case "$1" in
  e100) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqf_trunk512_oc100 $SQF --num_epochs 100 ;;
  wide) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqf_d256x5_trunk512_oc50 $SQF --num_epochs 50 --sqf_d_model 256 --sqf_depth 5 ;;
  trunk1024) step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqf_trunk1024_oc50 $SQF --num_epochs 50 --tsconv_backbone_dim 1024 ;;
esac
