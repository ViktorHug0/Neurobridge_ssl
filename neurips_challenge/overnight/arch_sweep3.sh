#!/bin/bash
# Crossovers on scratch LaBraM (LAB recipe): native 120 Hz, learned spatial mix -> virtual electrode
# tokens, TSConv band-power branch in the head. usage: arch_sweep3.sh <case>
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
LAB="--learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 --ema_decay 0.9998 --ema_warmup --eeg_aug --eeg_aug_type channel_dropout"
S="bash neurips_challenge/overnight/fm.sh LaBraM"
case "$1" in
  0) step $S scratch_native120 --fm_no_pretrained $LAB --fm_arch_kwargs '{"n_times":120,"patch_size":120}' ;;
  1) step $S scratch_virt32 --fm_no_pretrained $LAB --fm_arch_kwargs '{"virtual_channels":32}' ;;
  2) step $S scratch_power --fm_no_pretrained $LAB --fm_arch_kwargs '{"power_branch":true}' ;;
  3) step $S scratch_virt63 --fm_no_pretrained $LAB --fm_arch_kwargs '{"virtual_channels":63}' ;;
  4) step env BATCH=512 $S scratch_b512 --fm_no_pretrained $LAB ;;
  5) step env BATCH=64 $S scratch_b64 --fm_no_pretrained $LAB ;;
  6) step $S scratch_stop10 --fm_no_pretrained ${LAB/--stop_after_epoch 8/--stop_after_epoch 10} ;;
  10) step $S scratch_stop12 --fm_no_pretrained ${LAB/--stop_after_epoch 8/--stop_after_epoch 12} ;;
  7) step env BATCH=64 $S scratch_p40_b64 --fm_no_pretrained $LAB --fm_arch_kwargs '{"patch_size":40,"conv_out_channels":40}' ;;
  8) step $S scratch_d18 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":18}' ;;
  9) step $S scratch_d24 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":24}' ;;
  11) step $S scratch_w600 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":24,"num_heads":12}' ;;
  12) step $S scratch_w400_stop10 --fm_no_pretrained ${LAB/--stop_after_epoch 8/--stop_after_epoch 10} --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8}' ;;
  13) step $S scratch_w400_d18 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8,"num_layers":18}' ;;
  14) step $S scratch_w400_kdtf5 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8}' --distill_targets results/things_eeg/neurips_track1/distill/teacher_tf5.npy --distill_lambda 1.0 --distill_tau 0.005 ;;
  15) step $S scratch_d32 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":32}' ;;
  16) step $S scratch_w400_d24 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8,"num_layers":24}' ;;
  17) step $S scratch_w400d18_kdtf5 --fm_no_pretrained $LAB --fm_arch_kwargs '{"conv_out_channels":16,"num_heads":8,"num_layers":18}' --distill_targets results/things_eeg/neurips_track1/distill/teacher_tf5.npy --distill_lambda 1.0 --distill_tau 0.005 ;;
  18) step $S scratch_d48 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":48}' ;;
  19) step $S scratch_d32_kdtf5 --fm_no_pretrained $LAB --fm_arch_kwargs '{"num_layers":32}' --distill_targets results/things_eeg/neurips_track1/distill/teacher_tf5.npy --distill_lambda 1.0 --distill_tau 0.005 ;;
esac
