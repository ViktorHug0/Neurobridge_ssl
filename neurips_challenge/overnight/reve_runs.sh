#!/bin/bash
# REVE fine-tuning plan (2026-09-23). usage: reve_runs.sh r1|r123
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface HF_HUB_OFFLINE=1
IN120="--eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench --time_window 0 120"
case "$1" in
  # R1: train on the grader's own input (official 120 Hz, interpolated to 200 in the wrapper).
  r1)   step env OUT_DIR=./results/things_eeg/neurips_track1/reve/r1_in120 \
          bash neurips_challenge/run_reve_repro.sh $IN120 ;;
  # R1+R2+R3: + head lr 1e-3 / backbone 1e-4, 1 head-only epoch, REVE flatten head.
  r123) step env OUT_DIR=./results/things_eeg/neurips_track1/reve/r123_in120_lr_head \
          bash neurips_challenge/run_reve_repro.sh $IN120 --learning_rate 1e-3 \
          --fm_backbone_lr_scale 0.1 --fm_head_warmup_epochs 1 --fm_head flatten --fm_head_dropout 0.1 ;;
  # R1+R2: the lr split and head warmup with the original mean head, to isolate R3's head.
  r12)  step env OUT_DIR=./results/things_eeg/neurips_track1/reve/r12_in120_lr \
          bash neurips_challenge/run_reve_repro.sh $IN120 --learning_rate 1e-3 \
          --fm_backbone_lr_scale 0.1 --fm_head_warmup_epochs 1 ;;
esac
