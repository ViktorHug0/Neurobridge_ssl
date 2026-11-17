#!/bin/bash
# one ST-EEGFormer (pretrained small, braindecode/STEEGFormer-small) clean-proxy run from a queue line; later flags override the recipe.
#   job_stf.sh <name> <fold> <extra train.py flags...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2; shift 2
DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K gen --eeg_encoder_type STEEGFormer --batch_size 128 --learning_rate 2e-4 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5 --save_epochs 6 8 --things_lag 5 --time_jitter 3 "$@"
