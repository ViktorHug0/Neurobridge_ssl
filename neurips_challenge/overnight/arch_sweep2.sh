#!/bin/bash
# Scratch LaBraM anatomy on its own recipe (LAB): tokenization, depth, width, pooling; seed + KD controls.
# usage: arch_sweep2.sh <case>. LaBraM here = one 200-sample patch per channel -> 63 channel tokens.
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
LAB="--learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 --ema_decay 0.9998 --ema_warmup --eeg_aug --eeg_aug_type channel_dropout"
S="bash neurips_challenge/overnight/fm.sh LaBraM"
case "$1" in
  0) step env SEED=1 $S scratch_seed1 --fm_no_pretrained $LAB ;;
  1) step $S scratch_p40 --fm_no_pretrained $LAB --fm_arch_kwargs '{"patch_size":40,"conv_out_channels":40}' ;;
  2) step $S scratch_p100 --fm_no_pretrained $LAB --fm_arch_kwargs '{"patch_size":100,"conv_out_channels":16}' ;;
  3) step $S scratch_d6 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":6}' ;;
  4) step $S scratch_w400 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8}' ;;
  5) step $S scratch_lintok --fm_no_pretrained $LAB --fm_arch_kwargs '{"neural_tokenizer":false}' ;;
  6) step $S scratch_meanpool --fm_no_pretrained $LAB --fm_arch_kwargs '{"use_mean_pooling":true}' ;;
  7) step $S scratch_d4 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":4}' ;;
  8) step $S scratch_kdv2avg --fm_no_pretrained $LAB --distill_targets results/things_eeg/neurips_track1/distill/teacher_v2avg.npy --distill_lambda 1.0 --distill_tau 0.005 ;;
esac
