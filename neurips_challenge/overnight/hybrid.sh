#!/bin/bash
# Squeezeformer+power / FBTN hybrid (one scratch model). usage: hybrid.sh <tag> [extra train.py args]
source "$(dirname "$0")/common.sh"
TAG=$1; shift
step env AVERAGE= bash $R SqueezeTangentHybrid "$TAG" --grad_clip 0 --early_stop_patience 0 --num_epochs 50 "$@"
