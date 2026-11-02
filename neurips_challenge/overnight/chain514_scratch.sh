#!/bin/bash
# 514 queue: fine-tune the denoising-pretrained Squeezeformer, then LaBraM's architecture from scratch.
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqfp_pre_loo --tsconv_pool_kernel 25 --tsconv_pool_stride 2 \
    --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch \
    --init_encoder results/things_eeg/neurips_track1/pretrain/sqfp_loo_m30.pth
# Curriculum: learn on 4-trial averages (where scratch models are strong), then single trials.
step bash $R OrthoFastTSConvSqueezeformer sqfp_avgA --tsconv_pool_kernel 25 --tsconv_pool_stride 2 \
    --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch
A=$(ls results/things_eeg/neurips_track1/recipe/sqfp_avgA/*/*/checkpoint_test_best.pth | tail -1)
step env AVERAGE= bash $R OrthoFastTSConvSqueezeformer sqfp_avg2single --tsconv_pool_kernel 25 --tsconv_pool_stride 2 \
    --tsconv_backbone_dim 512 --grad_clip 0 --early_stop_patience 0 --num_epochs 50 --tsconv_power_branch \
    --init_encoder "$A"
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
step bash neurips_challenge/overnight/fm.sh LaBraM scratch_full_lr1e-3 --fm_no_pretrained --learning_rate 1e-3 \
    --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 \
    --ema_decay 0.9998 --ema_warmup --eeg_aug --eeg_aug_type channel_dropout
