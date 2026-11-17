#!/bin/bash
# score_one.sh <recipe> <fold>: scorer.sh's new-family read for ONE run, to run on a GPU node with free memory (srun --jobid=<job> --overlap)
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi; L=../../logs/mi/r11; r=$1; k=$2; [ -e $L/${r}_f$k.out ] && exit; export BS=${BS:-128}
B=mi_d6_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth,mi_rmix_ho5k_lag5_jit3_soft3_checkpoint_ep10.pth,mi_plab_ho5k_lag5_long_soft3_checkpoint_ep10.pth
S=$B,mi_stf_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth; P="../../.venv/bin/python nv02_union_screen.py"
REF=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth TWINS=9:x RMIX=_jit3 SAVE=$r SWAP=5:$r:checkpoint_test_best.pth ../../.venv/bin/python nu04_stack.py $k > $L/${r}_f$k.tmp1 2>&1 || { tail -3 $L/${r}_f$k.tmp1; exit 1; }
{ grep "reference six\|swap\|in the six" $L/${r}_f$k.tmp1 | cut -c1-240
  echo "7th next to the six:"; $P $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
  echo "10th next to B:"; BASE_ADD=$B $P $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
  echo "11th next to sub24 (B + ST soft):"; BASE_ADD=$S $P $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
  echo "swapped into each slot of sub24 (row = member replaced):"; SWAPIN=${r}_checkpoint_test_best.pth BASE_ADD=$S $P $k 2>&1 | grep "^mi_" | cut -c1-175; } > $L/${r}_f$k.tmp2
command mv $L/${r}_f$k.tmp2 $L/${r}_f$k.out; command rm -f $L/${r}_f$k.tmp1
