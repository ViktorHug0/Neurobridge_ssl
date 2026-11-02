#!/bin/bash
# Foundation-model full fine-tune on the official 120 Hz input. usage: fm.sh <LaBraM|EEGPT|REVE> <tag> [extra args]
source "$(dirname "$0")/common.sh"
ENC=$1; TAG=$2; shift 2
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
step env OUT_DIR=./results/things_eeg/neurips_track1/${ENC,,}/$TAG bash neurips_challenge/run_reve_repro.sh \
    --eeg_encoder_type $ENC --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --time_window 0 120 --amp_dtype bfloat16 "$@"
