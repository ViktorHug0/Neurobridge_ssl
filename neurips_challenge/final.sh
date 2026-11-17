#!/bin/bash
# Final sealed-phase member (README "Final sealed model"): a cv4 recipe retrained on all 20 Alljoined people + sub-21..30,
# stopped at its cv4 epoch budget (best epoch of the 4-fold sealed curve), plus 10 of the 80 test reps of every person
# (every 8th rep, all sessions; user call 2026-11-05). Subject 1 is only an in-sample smoke score.
#   usage: final.sh <cv4 recipe> <epoch budget> [tag [extra train.py flags, e.g. --save_epochs 10]]   -> results/alljoined/track1/final/<recipe>[_<tag>]/<arm>/<run>/
#   (a tag gives a variant final, e.g. another budget, its own directory: train.py refuses to reuse one)
cd "$(dirname "$0")/.."
SPEC=$(bash neurips_challenge/cv4.sh --list | grep "^$1:" | sed 's/^[^:]*: *//') && [ -n "$SPEC" ] || { echo "unknown recipe $1"; exit 1; }
ARM=${SPEC%% *}; FLAGS="${SPEC#* }"
SPLIT=final/$1${3:+_$3} TRAIN_SUBJECTS="$(seq -s ' ' 1 30)" TEST_SUBJECTS=1 bash neurips_challenge/alljoined_run.sh "$ARM" $FLAGS \
  --select_best_on fixed --stop_after_epoch "$2" --train_test_reps 10 "${@:4}"; exit
