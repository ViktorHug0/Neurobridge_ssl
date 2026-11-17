#!/bin/bash
# wait_run.sh <name>_f<k>: block until that nc_train run has written its sealed score; exit 1 if its log stalls for 15 min
L=/nasbrain/p20fores/Neurobridge_SSL/logs/mi/train_$1.log
until grep -aq "track1_score_sealed.json" $L; do
  [ $(( $(date +%s) - $(stat -c %Y $L) )) -gt 900 ] && { echo "STALLED $1"; exit 1; }
  sleep 60
done
