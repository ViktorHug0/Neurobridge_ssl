#!/bin/bash
# Testbed for the sealed cohort size (11 unseen participants, tracks.html): 2 splits of 11 test / 9 train
# Alljoined subjects (+ THINGS-EEG2 for *_things recipes), same recipes and flags as cv4.sh. Used to tune the
# predict()-time batch transform with 11 participants mixed per batch (ROUTING_LOG E27+). Not a model-selection
# protocol: models see 9 subjects, not 15.
#   usage: s11.sh <cv4 recipe> [split ...]     (default: 1 2)
# Results: results/alljoined/track1/s11/<recipe>/fold<k>/<arm>/<run>/
cd "$(dirname "$0")/.."
FOLDS=("" "2 3 4 5 7 11 14 15 16 17 20" "1 2 3 6 8 9 10 12 13 18 19")
NAME=$1; shift
SPEC=$(bash neurips_challenge/cv4.sh --list | grep "^$NAME: " | cut -d' ' -f2-); [ -n "$SPEC" ] || { echo "unknown recipe $NAME"; exit 1; }
ARM=${SPEC%% *} FLAGS="${SPEC#* }"
for K in ${@:-1 2}; do
  TEST=${FOLDS[$K]}; OUT=results/alljoined/track1/s11/$NAME/fold$K/$ARM
  if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip $NAME split$K (scored)"; continue; fi
  TRAIN=""; for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || TRAIN+="$s "; done
  [[ $NAME == *_things ]] && TRAIN+="$(seq -s ' ' 21 30)"
  echo "=== $(date '+%F %T') $NAME split$K  train: $TRAIN test: $TEST"
  SPLIT=s11/$NAME/fold$K TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh "$ARM" \
    --select_best_on fixed --val_concept_ratio 0 --sealed_test_every 1 --sealed_test_fraction 0.5 $FLAGS \
    || echo "FAILED $NAME split$K"
done
