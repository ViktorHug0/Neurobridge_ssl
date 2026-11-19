#!/bin/bash
# read_swap.sh <recipe> <fold> <member of E2 to replace, recipe:checkpoint> <checkpoint> ...: wait for a screen to finish
# (track1_score_sealed.json), cache the listed checkpoints, then the aggregation gain (ny11 SOLO) and each swap in E2 (ny12).
cd /nasbrain/p20fores/Neurobridge_SSL; N=$1 K=$2 T=$3; shift 3; R=results/alljoined/track1/cv4/$N/fold$K O=logs/mi/read_${N}_f$K.out
until ls $R/*/*/track1_score_sealed.json >/dev/null 2>&1; do sleep 60; done; sleep 30
.venv/bin/python scripts/mi/nu09_cache.py $K $(for c in "$@"; do echo $N:$c; done) > $O 2>&1
SOLO=1 .venv/bin/python scripts/mi/ny11_agg_gain.py $K $(for c in "$@"; do echo new@${c%.pth}=$N:$c; done) >> $O 2>&1
.venv/bin/python scripts/mi/ny12_add_one.py $K $(for c in "$@"; do echo "$T>$N:$c"; done) >> $O 2>&1
echo READ DONE >> $O
