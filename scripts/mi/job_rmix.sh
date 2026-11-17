#!/bin/bash
# sbatch job: one rmix (TokenREVE) clean-proxy run.   job_rmix.sh <name> <fold> <extra train.py flags...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2; shift 2
DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K gen --eeg_encoder_type TokenREVE --arch_kwargs '{"tokens":"mixed","registers":4}' --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5 --save_epochs 8 10 "$@"
