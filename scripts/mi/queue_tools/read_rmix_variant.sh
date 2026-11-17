#!/bin/bash
# read_rmix_variant.sh <recipe> <fold> <job>: once scorer.sh has cached the run's outputs, the variant in sub24's ten at 5k / 14k / 40k:
# swapped for its in-batch soft twin (rmix soft), for the plain rmix, and added as an 11th member -> logs/mi/r12_<recipe>_f<k>.out
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi; r=$1; k=$2; J=${3:-76146}; c=${r}_checkpoint_ep10.pth
B=mi_d6_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth,mi_rmix_ho5k_lag5_jit3_soft3_checkpoint_ep10.pth,mi_plab_ho5k_lag5_long_soft3_checkpoint_ep10.pth,mi_stf_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth
until [ -e ../../results/things_eeg/mi/nc/u01_f${k}_$c.npy ] && [ -e ../../logs/mi/r11/${r}_f$k.out ]; do sleep 30; done
{ cat ../../logs/mi/r11/${r}_f$k.out | grep "alone vs\|in the six" | cut -c1-200
  echo "swapped into sub24's ten (rows: for rmix soft, for rmix):"; OPEN=1 SWAPIN=$c BASE_ADD=$B srun --jobid=$J --overlap --export=ALL ../../.venv/bin/python nv02_union_screen.py $k 2>&1 | grep "^mi_rmix" | cut -c1-260
  echo "11th next to sub24:"; OPEN=1 BASE_ADD=$B srun --jobid=$J --overlap --export=ALL ../../.venv/bin/python nv02_union_screen.py $k ${r}_checkpoint_ep10 2>&1 | grep "^mi_" | cut -c1-260; } > ../../logs/mi/r12_${r}_f$k.out 2>&1
