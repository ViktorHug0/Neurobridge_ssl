#!/bin/bash
# Where does scratch LaBraM's 0.4651 come from? Recipe x architecture crossovers. usage: arch_sweep.sh <case>
# LAB = scratch LaBraM's recipe (batch 128, 20-ep onecycle stopped at 8, EMA, channel dropout, full data).
source "$(dirname "$0")/common.sh"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
LAB="--learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 --ema_decay 0.9998 --ema_warmup --eeg_aug --eeg_aug_type channel_dropout"
FULL="--val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0"
SQF="--tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --tsconv_power_branch --grad_clip 0"
FM="bash neurips_challenge/overnight/fm.sh"
case "$1" in
  # recipe -> scratch models (is it the recipe?)
  0) step env AVERAGE= BATCH=128 bash $R OrthoFastTSConvSqueezeformer sqfp_labrecipe $SQF $LAB ;;
  1) step env AVERAGE= BATCH=128 bash $R FilterBankTangentNet fbtn_labrecipe --grad_clip 0 --arch_kwargs {} $LAB ;;
  2) step env AVERAGE= BATCH=128 bash $R ATM atm_labrecipe --atm_no_subject_token --atm_backbone_dim 1024 $LAB ;;
  3) step env AVERAGE= BATCH=128 bash $R TSConv_parameterizable t2_labrecipe --tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512 --tsconv_power_branch $LAB ;;
  # architecture -> scratch recipe (is it the architecture?)
  4) step $FM LaBraM scratch_sqfrecipe --fm_no_pretrained --learning_rate 1e-3 --num_epochs 50 $FULL ;;
  # LaBraM recipe, one factor removed
  5) step $FM LaBraM scratch_noema --fm_no_pretrained ${LAB/--ema_decay 0.9998 --ema_warmup/} ;;
  6) step $FM LaBraM scratch_nocd --fm_no_pretrained ${LAB/--eeg_aug --eeg_aug_type channel_dropout/} ;;
  7) step env BATCH=1024 $FM LaBraM scratch_b1024 --fm_no_pretrained $LAB ;;
  8) step $FM LaBraM scratch_nostop --fm_no_pretrained ${LAB/--stop_after_epoch 8/} ;;
  # other transformer families from scratch
  9) step $FM REVE scratch_lab --fm_no_pretrained --reve_input_normalization none $LAB ;;
esac
