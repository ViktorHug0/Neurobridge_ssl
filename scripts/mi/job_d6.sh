#!/bin/bash
# sbatch job: one REVE-d6 clean-proxy run.   job_d6.sh <name> <fold> <extra train.py flags...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 K=$2; shift 2
DIRECT=1 bash scripts/mi/nc_train.sh $NAME $K reve --fm_no_pretrained --fm_arch_kwargs '{"depth":6}' --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --op_transfer_prob 0.5 --save_epochs 8 10 "$@"
# 2026-11-11 20:35: the interactive job was cancelled and the Claude session now lives inside job 73733 (518): keep the job (and the session) alive after the run; scancel 73733 when done
[ "$SLURM_JOB_ID" = 73733 ] && sleep 2d
