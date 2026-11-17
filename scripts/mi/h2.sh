#!/bin/bash
# Q09 second-proxy rmix run (held-out images drawn outside the ds007964 images): h2.sh <fold> [ds]
#   -> mi_rmix_h2_lag5 (reference) or mi_rmix_h2_lag5_ds2 (with the cohort, x2); JOB= / DIRECT=1 as nc_train.sh
cd /nasbrain/p20fores/Neurobridge_SSL; D=/nasbrain/p20fores/ds007964
[ "$2" = ds ] && N=mi_rmix_h2_lag5_ds2 X="--extra_trials_dir $D --extra_shift 4 --extra_repeat 2" || N=mi_rmix_h2_lag5 X=""
bash scripts/mi/nc_train.sh $N $1 gen --eeg_encoder_type TokenREVE --arch_kwargs '{"tokens":"mixed","registers":4}' --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5 --things_lag 5 --save_epochs 8 10 --holdout_avoid $D/targets.npz $X
