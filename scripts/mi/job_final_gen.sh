#!/bin/bash
# a final member of the `gen` arm whose recipe is not in cv4.sh (LaBraM): job_final_gen.sh <final name> <epoch budget> <train.py flags...>
# = final.sh's call: all 20 Alljoined people + sub-21..30, 10 of the 80 test repeats, stopped at the budget -> results/alljoined/track1/final/<name>/gen/
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1 B=$2; shift 2
SPLIT=final/$NAME TRAIN_SUBJECTS="$(seq -s ' ' 1 30)" TEST_SUBJECTS=1 bash neurips_challenge/alljoined_run.sh gen "$@" \
  --select_best_on fixed --stop_after_epoch "$B" --train_test_reps 10 > logs/mi/final_$NAME.log 2>&1
