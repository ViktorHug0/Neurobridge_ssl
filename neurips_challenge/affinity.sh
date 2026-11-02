#!/bin/bash
# Subject-affinity "datamodel": scratch REVE-6 (no THINGS) on each of 40 balanced random 5-subject sets
# (affinity_subsets.txt), scored per epoch-pair on the 15 Alljoined subjects it did not train on. Regressing each
# target's score on which subjects were in the training set gives source->target affinity at fixed data size.
# Budget = the scaling experiment's at N = 5 (same optimizer steps as cv4's 15-subject recipe, early stop).
# Claim-walked by alljoined_run.sh: run it on several GPUs at once. Results: results/alljoined/track1/affinity/s<i>/reve/
cd "$(dirname "$0")/.."
FLAGS="--fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --reve_input_normalization trial_zscore"
i=0
grep -v '^#' neurips_challenge/affinity_subsets.txt | while read -r TRAIN; do
  i=$((i + 1))
  OUT=results/alljoined/track1/affinity/s$i/reve
  if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip s$i (scored)"; continue; fi
  TEST=$(for s in $(seq 1 20); do [[ " $TRAIN " == *" $s "* ]] || echo -n "$s "; done)
  echo "=== $(date '+%F %T') affinity s$i train: $TRAIN test: $TEST"
  SPLIT=affinity/s$i TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh reve \
    --num_epochs 60 --stop_after_epoch 30 --select_best_on fixed --val_concept_ratio 0 \
    --sealed_test_every 2 --sealed_test_fraction 0.5 --sealed_early_stop_patience 3 $FLAGS < /dev/null \
    || echo "FAILED affinity s$i"
done
echo "done affinity"
