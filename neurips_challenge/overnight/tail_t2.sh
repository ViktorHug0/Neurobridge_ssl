#!/bin/bash
# Runs after chain_t2_b. 200 Hz native training (REVE's cache), kernels held at T2's durations in
# ms; scored on the official 120 Hz input, which score_track1.py interpolates to 200 samples.
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R TSConv_parameterizable t2_200hz --learning_rate 1e-3 $SINGLE \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_200Hz_reve \
    --time_window 0 200 --tsconv_temporal_kernel 42 --tsconv_pool_kernel 42 --tsconv_pool_stride 3
