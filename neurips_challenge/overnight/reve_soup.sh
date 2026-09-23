#!/bin/bash
# Overnight REVE set (2026-09-23): the 10-epoch R1 recipe (0.4074) with ONE change per run.
# All runs keep --seed 33 (identical head init) so they stay weight-soup compatible;
# --train_rng_seed only re-seeds batch order and dropout. usage: reve_soup.sh <name>
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface HF_HUB_OFFLINE=1
case "$1" in
  s1)       X="--train_rng_seed 1" ;;
  s2)       X="--train_rng_seed 2" ;;
  lr5e-5)   X="--learning_rate 5e-5" ;;
  lr2e-4)   X="--learning_rate 2e-4" ;;
  wd0.1)    X="--weight_decay 0.1" ;;
  last16)   X="--fm_unfreeze_last_k 16" ;;
  e7)       X="--num_epochs 7" ;;
  bag1)     X="--val_concept_seed 1" ;;
  wsd)      X="--lr_scheduler wsd --num_epochs 6 --onecycle_pct_start 0.0833 --wsd_decay_start 0.5" ;;
  llrd0.8)  X="--fm_layer_decay 0.8" ;;
  *) echo "unknown run $1"; exit 2 ;;
esac
step env OUT_DIR=./results/things_eeg/neurips_track1/reve/soup_$1 \
    bash neurips_challenge/run_reve_repro.sh \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench \
    --time_window 0 120 --amp_dtype bfloat16 --num_epochs 10 --seed 33 $X
