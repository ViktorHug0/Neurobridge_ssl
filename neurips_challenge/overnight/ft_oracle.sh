#!/bin/bash
# SUBJECT_LOG D4, the intra-subject ceiling: for each held-out subject of cv4 fold K, start from the fold's model (which
# never saw the subject) and fine-tune it on the subject's OWN training trials (other images), scoring the subject's
# test trials every epoch. Dispatched by cv4.sh for recipes ftor_* so it runs from the align queue between screens.
#   ft_oracle.sh <ftor_tsc|ftor_reve> <fold>
cd "$(dirname "$0")/../.."
R=$1; K=$2
FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")
case $R in
  ftor_tsc)  BASE=tsc_dbn40_bn64_things; ARM=tsc
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs 6 --learning_rate 1e-4" ;;
  ftor_reve) BASE=reve_d6_things; ARM=reve
             ARGS="--fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --reve_input_normalization trial_zscore --num_epochs 6 --stop_after_epoch 6 --learning_rate 1e-4" ;;
  *) echo "unknown $R"; exit 1 ;;
esac
CKPT=$(ls results/alljoined/track1/cv4/$BASE/fold$K/*/2026*/checkpoint_sealed_best.pth | head -1)
OUT=results/alljoined/track1/cv4/$R/fold$K
# entry-level claim read by align_dyn.sh (<first subdir>/.claim/job): one worker per oracle entry
mkdir -p $OUT/run/.claim && echo "${SLURM_JOB_ID:-$$}" > $OUT/run/.claim/job
for s in ${FOLDS[$K]}; do
  ls $OUT/s$s/$ARM/2026*/track1_score.json >/dev/null 2>&1 && continue
  echo "=== $(date '+%F %T') $R fold$K subject $s from $CKPT"
  SPLIT=cv4/$R/fold$K/s$s TRAIN_SUBJECTS=$s TEST_SUBJECTS=$s bash neurips_challenge/alljoined_run.sh $ARM $ARGS \
    --init_checkpoint "$CKPT" --select_best_on fixed --val_concept_ratio 0 --sealed_test_every 1 --sealed_test_fraction 0.5
done
# align_dyn.sh treats "<first subdir>/2026*/track1_score.json" as done: a marker that sorts first
n=$(ls $OUT/s*/$ARM/2026*/track1_score.json 2>/dev/null | wc -l)
[ "$n" -eq 5 ] && mkdir -p $OUT/a_done/2026-done && touch $OUT/a_done/2026-done/track1_score.json
echo "done $R fold$K ($n/5 subjects)"
