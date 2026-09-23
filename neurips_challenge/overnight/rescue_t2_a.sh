#!/bin/bash
# Runs after chain_t2_a exits; re-runs any of its steps that did not produce a score.
source "$(dirname "$0")/common.sh"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
scored t2_boot2 || step env AVERAGE= bash $R TSConv_parameterizable t2_boot2 $T2 $SHORT $BOOT
scored t2_avg4  || step              bash $R TSConv_parameterizable t2_avg4  $T2 $SHORT
scored t2_crop  || step env AVERAGE= bash $R TSConv_parameterizable t2_crop  $T2 $SINGLE --time_window 24 120
