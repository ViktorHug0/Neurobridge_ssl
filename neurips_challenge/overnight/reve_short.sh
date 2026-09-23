#!/bin/bash
# REVE R1 (0.4063) with the one-cycle decay inside the epoch-5-8 peak: the 40-epoch runs
# peaked before any decay. bf16, official 120 Hz input, one lr for every layer.
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface HF_HUB_OFFLINE=1
step env OUT_DIR=./results/things_eeg/neurips_track1/reve/r1_e10_bf16 \
    bash neurips_challenge/run_reve_repro.sh \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --time_window 0 120 --amp_dtype bfloat16 --num_epochs 10
