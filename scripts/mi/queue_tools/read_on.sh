#!/bin/bash
# read_on.sh <holder job id> <recipe> <fold> <E2 member to replace, recipe:checkpoint> <checkpoint> ...: wait for a screen to finish, then ON
# THAT JOB'S GPU (never on 515 next to a large training): cache the checkpoints, swap each into E2 and add each to E2 (ny12).
cd /nasbrain/p20fores/Neurobridge_SSL; J=$1 N=$2 K=$3 T=$4; shift 4; R=results/alljoined/track1/cv4/$N/fold$K O=logs/mi/read_${N}_f$K.out
until ls $R/*/*/track1_score_sealed.json >/dev/null 2>&1; do sleep 60; done; sleep 30
S="srun --jobid=$J --overlap --export=ALL"
$S .venv/bin/python scripts/mi/nu09_cache.py $K $(for c in "$@"; do echo $N:$c; done) > $O 2>&1
$S .venv/bin/python scripts/mi/ny12_add_one.py $K $(for c in "$@"; do echo "$T>$N:$c"; done) $(for c in "$@"; do echo "$N:$c"; done) >> $O 2>&1
echo READ DONE >> $O
