#!/bin/bash
# one clean-proxy (--holdout_images 5000) cv4 run inside the mi_gpu job (or directly, DIRECT=1, e.g. under sbatch), outside cv4.sh's recipe table:
#   nc_train.sh <name> <fold> <arm> <train.py flags...>   -> results/alljoined/track1/cv4/<name>/fold<k>/<arm>/
cd /nasbrain/p20fores/Neurobridge_SSL
FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")
NAME=$1 K=$2 ARM=$3; shift 3
TEST=${FOLDS[$K]}; TRAIN=""; for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || TRAIN+="$s "; done; TRAIN+="$(seq -s ' ' 21 30)"
[ -n "$ALLJ" ] && TRAIN="$ALLJ $(seq -s " " 21 30)"  # ALLJ="1 6 8 ...": a subset of the fold's Alljoined training people (data-size checks)
J=${JOB:-$(squeue -u $USER -h -n mi_gpu -o %i | head -1)}
[ -n "$DIRECT" ] && RUN="" || RUN="srun --jobid=$J --overlap"
SPLIT=cv4/$NAME/fold$K TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" $RUN bash neurips_challenge/alljoined_run.sh "$ARM" \
  --select_best_on fixed --sealed_test_every 1 --sealed_test_fraction 0.5 --holdout_images 5000 "$@" > logs/mi/train_${NAME}_f$K.log 2>&1
