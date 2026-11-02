#!/bin/bash
# Subject-scaling curve: how much does each extra training subject help on NEW subjects?
# Test sets: cv4 folds 1 and 2. Training sets: the first N of a seeded permutation of the fold's 15 training
# subjects (nested, so each N adds subjects to the previous set), N = 1 3 5 ... 13; N = 15 is the cv4 run.
# Same optimizer-step budget for every N as the cv4 recipe at 15 subjects (epochs x 15/N, same schedule
# shape), per-epoch sealed scoring on ~15 evaluations, early stop after 5 evaluations without improvement;
# a point is the run's best evaluation (neurips_challenge/scaling_summary.py).
# THINGS=first|last adds the 10 THINGS-EEG2 subjects (21-30, seeded order) before / after the Alljoined order,
# N = 1, 3 ... 25. With THINGS=last, N <= 15 is the plain experiment (reused); the 25-subject set is shared.
#   usage: [ORDER_SEED=1] [THINGS=first|last] scaling.sh <tsc|plab|reve> [fold ...]      (DRY=1 prints the commands only)
# Results: results/alljoined/track1/scaling/<model>/fold<k>/o<seed>[_things<first|last>]/n<N>/<arm>/
cd "$(dirname "$0")/.."
FOLDS=("" "2 4 5 11 15" "7 14 16 17 20")   # = cv4.sh folds 1 and 2
MODEL=$1; shift
case "$MODEL" in
  tsc)  ARM=tsc;  E15=12; STOP15=0;  FLAGS="" ;;                     # cv4 recipe tsc
  plab) ARM=plab; E15=20; STOP15=10; FLAGS="--amp_gain_range 2" ;;   # cv4 recipe plab_gain_s10
  reve) ARM=reve; E15=20; STOP15=10; PATIENCE=3  # cv4 reve_scratch at depth 6 (= depth 22 on cv4, ~2x faster); patience 3 bounds the cost
        FLAGS="--fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --reve_input_normalization trial_zscore" ;;
  *) echo "usage: scaling.sh <tsc|plab|reve> [fold ...]"; exit 1 ;;
esac
# REVE: fold 1 only for now (halves the cost; the running chains still pass "1 2"). REVE_BOTH_FOLDS=1 lifts it.
[ $MODEL = reve ] && [ -z "$REVE_BOTH_FOLDS" ] && set -- 1
SEED=${ORDER_SEED:-1}
NS="1 3 5 7 9 11 13"
if [ -n "$THINGS" ]; then
  [[ $THINGS =~ ^(first|last)$ ]] || { echo "THINGS must be first or last"; exit 1; }
  [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/things_caches_ok ] \
    || { echo "THINGS-EEG2 training caches not built on the NAS yet"; exit 1; }
  NS="$(seq -s ' ' 1 2 25)"
elif [ $MODEL = reve ]; then
  NS="$NS 15"  # no cv4 run of this exact model to serve as N = 15
fi
for K in ${@:-1 2}; do
  TEST=${FOLDS[$K]}; [ -n "$TEST" ] || { echo "no fold $K"; exit 1; }
  POOL=$(for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || echo -n "$s "; done)
  ORDER=$(.venv/bin/python -c "import numpy as np; print(*np.random.default_rng($SEED * 100 + $K).permutation('$POOL'.split()))")
  if [ -n "$THINGS" ]; then
    TORDER=$(.venv/bin/python -c "import numpy as np; print(*np.random.default_rng($SEED * 100 + 50 + $K).permutation(range(21, 31)))")
    if [ $THINGS = first ]; then ORDER="$TORDER $ORDER"; else ORDER="$ORDER $TORDER"; fi
  fi
  for N in $NS; do
    BASE=o$SEED${THINGS:+_things$THINGS}
    if [ "$THINGS" = last ] && [ $N -le 15 ]; then continue; fi  # = the plain experiment (N = 15: the cv4 run)
    [ $N = 25 ] && BASE=o${SEED}_thingslast                         # all 25 subjects: one run serves both orders
    OUT=results/alljoined/track1/scaling/$MODEL/fold$K/$BASE/n$N/$ARM
    if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip $OUT (scored)"; continue; fi
    TRAIN=$(echo $ORDER | cut -d' ' -f1-$N)
    EPOCHS=$(( (E15 * 15 + N / 2) / N )) EVERY=$(( (15 + N / 2) / N )); [ $EVERY -lt 1 ] && EVERY=1
    STOP=""; [ $STOP15 -gt 0 ] && STOP="--stop_after_epoch $(( (STOP15 * 15 + N / 2) / N ))"
    # THINGS runs: at least twice the 15-subject epochs. Equal steps would give N = 25 only 7 TSConv epochs, and
    # tsc_things (25 subjects, 12 epochs) was still rising at its last one; the early stop trims the excess
    if [ -n "$THINGS" ] && [ $EPOCHS -lt $((2 * E15)) ]; then
      EPOCHS=$((2 * E15)); [ $STOP15 -gt 0 ] && STOP="--stop_after_epoch $((2 * STOP15))"
    fi
    echo "=== $(date '+%F %T') $MODEL${THINGS:+ things $THINGS} fold$K N=$N train: $TRAIN test: $TEST  epochs $EPOCHS $STOP eval every $EVERY"
    [ -n "$DRY" ] && continue
    SPLIT=scaling/$MODEL/fold$K/$BASE/n$N TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh "$ARM" \
      --num_epochs $EPOCHS $STOP --select_best_on fixed --val_concept_ratio 0 \
      --sealed_test_every $EVERY --sealed_test_fraction 0.5 --sealed_early_stop_patience ${PATIENCE:-5} $FLAGS \
      || echo "FAILED $MODEL fold$K N=$N"
  done
done
echo "done scaling $MODEL folds ${@:-1 2}"
