#!/bin/bash
# Ensemble-distilled scratch student. usage: distill.sh <arch> <tag> <teacher.npy> <lambda> [extra train.py args]
source "$(dirname "$0")/common.sh"
ARCH=$1 TAG=$2 TEACHER=$3 LAMBDA=$4; shift 4
SQF="--tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --tsconv_power_branch"
[ "$ARCH" = OrthoFastTSConvSqueezeformer ] || SQF=""
step env AVERAGE= bash $R "$ARCH" "$TAG" $SQF --grad_clip 0 --early_stop_patience 0 --num_epochs 50 \
    --distill_targets "$TEACHER" --distill_lambda "$LAMBDA" "$@"
