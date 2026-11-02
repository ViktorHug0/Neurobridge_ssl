#!/bin/bash
# Follow-ups on the 514 interactive GPU.
bash neurips_challenge/overnight/labram.sh full_ema_s8_chdrop_lr8e-4 --learning_rate 8e-4 --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 --ema_decay 0.9998 --ema_warmup --eeg_aug --eeg_aug_type channel_dropout
