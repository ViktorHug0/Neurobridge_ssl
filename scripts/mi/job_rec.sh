#!/bin/bash
# sbatch job: one recurrent-member (RecurrentEEG) clean-proxy run.   job_rec.sh <name> <fold> <extra train.py flags...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2; shift 2
DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K gen --eeg_encoder_type RecurrentEEG --arch_kwargs '{"boundary":21}' --num_epochs 40 --op_transfer_prob 0.5 --save_epochs 24 32 36 "$@"
