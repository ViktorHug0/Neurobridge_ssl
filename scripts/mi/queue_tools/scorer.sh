#!/bin/bash
# scores every finished round-11 run of folds 1-3 once (on the interactive GPU): its member swapped into the sub16 six of its fold
# (nu04_stack.py -> logs/mi/r11/<recipe>_f<k>.out). ST-EEGFormer runs go in the LaBraM slot against the jitter LaBraM and are also read
# as an extra member next to the six and next to B (nv02_union_screen.py, all triplets).
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi; L=../../logs/mi; mkdir -p $L/r11
B=mi_d6_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth,mi_rmix_ho5k_lag5_jit3_soft3_checkpoint_ep10.pth,mi_plab_ho5k_lag5_long_soft3_checkpoint_ep10.pth
while :; do
  for f in $(find $L -name 'train_mi_*_f[1-3].log' -newermt '2026-11-16 08:00'); do
    n=$(basename $f .log); n=${n#train_}; k=${n##*_f}; r=${n%_f*}; sw=$r; ref=""
    case $r in
      mi_stf*|mi_cbm*|mi_egp*|mi_prev*|mi_rec*|mi_plab[248]*|mi_tsc*|mi_luna*) export BS=128; slot=5; sw=$r:checkpoint_test_best.pth; ref=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth;;
      mi_rmix_ho5k_lag5_jit[35]*) slot=1;; mi_d6_ho5k_lag5_jit[35]*) slot=0;; *) continue;;
    esac
    [ -e $L/r11/${r}_f$k.out ] || [ -e $L/r11/${r}_f$k.err ] && continue
    grep -q track1_score_sealed.json $f || continue
    if REF=$ref TWINS=9:x RMIX=_jit3 SAVE=$r SWAP=$slot:$sw ../../.venv/bin/python nu04_stack.py $k > $L/r11/${r}_f$k.tmp 2>&1; then
      grep "reference six\|swap\|in the six" $L/r11/${r}_f$k.tmp | cut -c1-240 > $L/r11/${r}_f$k.out
      if [ $slot = 5 ]; then
        { echo "7th next to the six:"; ../../.venv/bin/python nv02_union_screen.py $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
          echo "10th next to B:"; BASE_ADD=$B ../../.venv/bin/python nv02_union_screen.py $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
          echo "11th next to sub24 (B + ST soft):"; BASE_ADD=$B,mi_stf_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth ../../.venv/bin/python nv02_union_screen.py $k ${r}_check 2>&1 | grep "^mi_" | cut -c1-230
          echo "swapped into each slot of sub24 (row = member replaced):"; SWAPIN=${r}_checkpoint_test_best.pth BASE_ADD=$B,mi_stf_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth ../../.venv/bin/python nv02_union_screen.py $k 2>&1 | grep "^mi_" | cut -c1-175; } >> $L/r11/${r}_f$k.out
      fi
    elif grep -q OutOfMemory $L/r11/${r}_f$k.tmp; then command rm $L/r11/${r}_f$k.tmp; else command mv $L/r11/${r}_f$k.tmp $L/r11/${r}_f$k.err; fi
  done
  sleep 300
done
