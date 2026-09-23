#!/bin/bash
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R TSConv_parameterizable t2_lr1e3 $T2 $SINGLE
step env AVERAGE= bash $R TSConv_parameterizable t2_boot2 $T2 $SHORT $BOOT
step              bash $R TSConv_parameterizable t2_avg4  $T2 $SHORT
step env AVERAGE= bash $R TSConv_parameterizable t2_crop  $T2 $SINGLE --time_window 24 120
