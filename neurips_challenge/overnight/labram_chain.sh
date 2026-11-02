#!/bin/bash
# LaBraM follow-ups on 517: token head, a data-order seed, a higher lr.
bash neurips_challenge/overnight/labram.sh flat_lr3e-4 --learning_rate 3e-4 --num_epochs 20 --fm_head flatten --fm_head_dropout 0.1
bash neurips_challenge/overnight/labram.sh lr3e-4_s1 --learning_rate 3e-4 --num_epochs 20 --train_rng_seed 1
bash neurips_challenge/overnight/labram.sh lr5e-4_llrd0.65 --learning_rate 5e-4 --num_epochs 20 --fm_layer_decay 0.65
bash neurips_challenge/overnight/labram.sh full_lr3e-4_e10 --learning_rate 3e-4 --num_epochs 10 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0
bash neurips_challenge/overnight/labram.sh ml_lr3e-4 --learning_rate 3e-4 --num_epochs 20 --fm_head multilayer --fm_head_dropout 0.1
bash neurips_challenge/overnight/labram.sh full_lr3e-4_e6 --learning_rate 3e-4 --num_epochs 6 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0
bash neurips_challenge/overnight/fm.sh REVE lr2e-4_ema --learning_rate 2e-4 --num_epochs 10 --ema_decay 0.9998 --ema_warmup
bash neurips_challenge/overnight/labram.sh lr3e-4_ema9999 --learning_rate 3e-4 --num_epochs 20 --ema_decay 0.9999 --ema_warmup
bash neurips_challenge/overnight/labram.sh full_ema_s8_lr5e-4 --learning_rate 5e-4 --num_epochs 20 --stop_after_epoch 8 --val_concept_ratio 0 --select_best_on fixed --early_stop_patience 0 --ema_decay 0.9998 --ema_warmup
