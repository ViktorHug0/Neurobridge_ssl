#!/bin/bash
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R TSConv_parameterizable t2_trunk1024 $T2 $SINGLE --tsconv_backbone_dim 1024
step env AVERAGE= bash $R TSConv_parameterizable t2_f80       $T2 $SINGLE --tsconv_temporal_filters 80
step env AVERAGE= bash $R TSConv_parameterizable t2_l2norm    $T2 $SINGLE --eeg_l2norm --t_learnable --init_temperature 0.07
