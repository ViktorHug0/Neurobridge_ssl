#!/bin/bash
# sbatch job: one REVE-scratch (full depth) clean-proxy run.   job_scr.sh <name> <fold> <extra train.py flags...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2; shift 2
DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --op_transfer_prob 0.5 --save_epochs 8 10 "$@"
