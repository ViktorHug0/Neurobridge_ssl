#!/bin/bash
# LaBraM full fine-tune on the official 120 Hz input (interpolated to 200 in the wrapper). usage: labram.sh <tag> [extra args]
source "$(dirname "$0")/common.sh"
TAG=$1; shift
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
step env OUT_DIR=./results/things_eeg/neurips_track1/labram/$TAG bash neurips_challenge/run_reve_repro.sh \
    --eeg_encoder_type LaBraM --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --time_window 0 120 --amp_dtype bfloat16 "$@"
