#!/bin/bash
# FilterBankTangentNet (second-order family). usage: fbtn.sh <tag> '<arch_kwargs json>' [extra train.py args]
source "$(dirname "$0")/common.sh"
TAG=$1; KW=$2; shift 2
step env AVERAGE= bash $R FilterBankTangentNet "$TAG" --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --arch_kwargs "$KW" "$@"
