#!/bin/bash
# one LUNA clean-proxy run from a queue line.   job_luna.sh <name> <fold> pre|scr <extra train.py flags...>
#   pre: pretrained LUNA-base (PulpBio/LUNA, 7 M: 4 queries per 40-sample patch), LaBraM's schedule
#   scr: from scratch, wider read of the electrodes (8 queries per 16-sample patch = 15 tokens x 512), the REVE-scratch schedule
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2 V=$3; shift 3
C="--eeg_encoder_type LUNA --batch_size 128 --num_epochs 20 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5"
if [ $V = pre ]; then DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K gen $C --learning_rate 5e-4 --stop_after_epoch 10 --save_epochs 6 8 "$@"
else DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K gen $C --fm_no_pretrained --fm_arch_kwargs '{"patch_size":16,"num_queries":8}' --learning_rate 1e-3 --stop_after_epoch 12 --save_epochs 8 10 "$@"; fi
