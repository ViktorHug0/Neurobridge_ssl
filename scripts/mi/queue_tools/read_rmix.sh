#!/bin/bash
# read_rmix.sh <recipe> <fold>: wait for an rmix screen (12 epochs), cache ep08 / ep10 / test_best, then its aggregation gain (ny11 SOLO)
# and its swap for rmix soft in the weighted nine (ny12). Output: logs/mi/read_<recipe>_f<fold>.out
cd /nasbrain/p20fores/Neurobridge_SSL; N=$1 K=$2 R=results/alljoined/track1/cv4/$1/fold$2 O=logs/mi/read_$1_f$2.out
until ls $R/*/*/checkpoint_ep10.pth >/dev/null 2>&1 && grep -q "Epoch \[12/" $R/*/*/train.log 2>/dev/null; do sleep 60; done; sleep 240
.venv/bin/python scripts/mi/nu09_cache.py $K $N:checkpoint_ep08.pth $N:checkpoint_ep10.pth $N:checkpoint_test_best.pth > $O 2>&1
SOLO=1 .venv/bin/python scripts/mi/ny11_agg_gain.py $K new@8=$N:checkpoint_ep08.pth new@10=$N:checkpoint_ep10.pth new@best=$N:checkpoint_test_best.pth >> $O 2>&1
S=mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth
.venv/bin/python scripts/mi/ny12_add_one.py $K "$S>$N:checkpoint_ep08.pth" "$S>$N:checkpoint_ep10.pth" "$S>$N:checkpoint_test_best.pth" >> $O 2>&1
echo READ DONE >> $O
