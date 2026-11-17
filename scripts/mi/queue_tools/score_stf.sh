#!/bin/bash
# score_stf.sh <recipe> <fold>: scorer.sh's ST-EEGFormer read, to run ON a GPU node with free memory (bs 512 does not fit next to a training)
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi; L=../../logs/mi/r11; r=$1; k=$2; [ -e $L/${r}_f$k.out ] && exit
B=mi_d6_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth,mi_rmix_ho5k_lag5_jit3_soft3_checkpoint_ep10.pth,mi_plab_ho5k_lag5_long_soft3_checkpoint_ep10.pth
REF=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth TWINS=9:x RMIX=_jit3 SAVE=$r SWAP=5:$r:checkpoint_test_best.pth ../../.venv/bin/python nu04_stack.py $k > $L/${r}_f$k.tmp 2>&1 || exit 1
{ grep "reference six\|swap\|in the six" $L/${r}_f$k.tmp | cut -c1-240
  echo "7th next to the six:"; ../../.venv/bin/python nv02_union_screen.py $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
  echo "10th next to B:"; BASE_ADD=$B ../../.venv/bin/python nv02_union_screen.py $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230; } > $L/${r}_f$k.tmp2
command mv $L/${r}_f$k.tmp2 $L/${r}_f$k.out; command rm -f $L/${r}_f$k.err $L/${r}_f$k.tmp
