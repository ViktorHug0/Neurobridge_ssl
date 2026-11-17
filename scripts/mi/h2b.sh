#!/bin/bash
# Q09 second-proxy runs (held-out images drawn outside the ds007964 images), JOB= / DIRECT=1 as nc_train.sh:  h2b.sh <fold> <variant>
#   rmix variants: bank (cohort x2, --extra_bank), ds1 (cohort x1, no oversampling), ds4 (cohort x4)   [reference and x2: h2.sh]
#   d6 / d6ds (REVE-d6 reference / cohort x2), sqfx / sqfxds (SQF-X reference / cohort x2)
#   g18 / g18bank: rmix + ds004018 (16 people, 200 own images, x1) on the FIRST proxy (no image overlap), in-batch per-cohort loss / bank of its 200 images
cd /nasbrain/p20fores/Neurobridge_SSL; D=/nasbrain/p20fores/ds007964; G=/nasbrain/p20fores/ds004018; K=$1; H="--things_lag 5 --op_transfer_prob 0.5 --holdout_avoid $D/targets.npz"; C="--extra_trials_dir $D --extra_shift 4"
rmix() { bash scripts/mi/nc_train.sh $1 $K gen --eeg_encoder_type TokenREVE --arch_kwargs '{"tokens":"mixed","registers":4}' --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --save_epochs 8 10 $H "${@:2}"; }
d6() { bash scripts/mi/nc_train.sh $1 $K reve --fm_no_pretrained --fm_arch_kwargs '{"depth":6}' --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 12 --save_epochs 8 10 $H "${@:2}"; }
sqfx() { bash scripts/mi/nc_train.sh $1 $K sqf --num_epochs 40 --tsconv_domain_bn_boundary 21 --save_epochs 32 36 $H "${@:2}"; }
case $2 in
  bank) rmix mi_rmix_h2_lag5_bank $C --extra_repeat 2 --extra_bank;;
  ds1) rmix mi_rmix_h2_lag5_ds1 $C;;
  s7) rmix mi_rmix_h2_lag5_s7 --seed 7;;  # the reference again with another seed: run-to-run noise of the proxy
  ds4) rmix mi_rmix_h2_lag5_ds4 $C --extra_repeat 4;;
  d6) d6 mi_d6_h2_lag5;;
  d6ds) d6 mi_d6_h2_lag5_ds2 $C --extra_repeat 2;;
  sqfx) sqfx mi_sqfx_h2_lag5;;
  sqfxds) sqfx mi_sqfx_h2_lag5_ds2 $C --extra_repeat 2;;
  rm50) H="--things_lag 5 --op_transfer_prob 0.5"; rmix mi_rmix_ho5k_lag5_rm50 --repeat_mix 0.5;;   # first proxy: repeat-averaging SNR augmentation
  rm100) H="--things_lag 5 --op_transfer_prob 0.5"; rmix mi_rmix_ho5k_lag5_rm100 --repeat_mix 1.0;;
  d6rm50) H="--things_lag 5 --op_transfer_prob 0.5"; d6 mi_d6_ho5k_lag5_rm50 --repeat_mix 0.5;;
  g18) H="--things_lag 5 --op_transfer_prob 0.5"; rmix mi_rmix_ho5k_lag5_g18 --extra_trials_dir $G --extra_shift 5;;
  g18bank) H="--things_lag 5 --op_transfer_prob 0.5"; rmix mi_rmix_ho5k_lag5_g18bank --extra_trials_dir $G --extra_shift 5 --extra_bank;;
esac
