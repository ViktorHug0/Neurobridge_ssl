#!/bin/bash
# The cross-subject evaluation protocol for the sealed phase (new subjects, new images): 4 fixed folds of 5
# Alljoined test subjects, training on the other 15 + the 10 THINGS-EEG2 subjects (sub-21..30,
# export_things_to_alljoined.py). Full description, rationale and decision rules: neurips_challenge/CV4_PROTOCOL.md.
# Do not edit FOLDS: every number in results/alljoined/track1/cv4/ is only comparable because the folds never change.
#
#   usage: cv4.sh <recipe> [fold ...]          one recipe on the given folds (default: 1 2 3 4)
#          cv4.sh --list                       print the recipes
# Results: results/alljoined/track1/cv4/<recipe>/fold<k>/<arm>/<run>/track1_score.json (per-subject sealed
# scores). Finished folds are skipped, and alljoined_run.sh claims each run, so several workers can be pointed
# at the same recipes and folds (overnight/align_dyn.sh).
cd "$(dirname "$0")/.."

FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")

# recipe -> "<alljoined_run.sh arm> <train.py flags>". Every recipe trains for a fixed budget and keeps its
# last epoch (--select_best_on fixed); the budgets come from the per-epoch sealed curves (CV4_PROTOCOL.md §5).
# The 5 *_optx_things recipes are the best ensemble (HISTORY.md: lean TTA 44.25 over 4 folds); the 5 plain
# *_things recipes are the same members without channel-operator transfer, kept as its baseline (44.02).
recipe() {
  case "$1" in
    reve_scratch_things)        echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10" ;;
    reve_d6_things)             echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10" ;;
    sqf_dbn_things)             echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;  # every BatchNorm per dataset (ids >= 21 = THINGS)
    tsc_dbn40_bn64_things)      echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    rmix_reg_things)            echo "gen  --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
    # + re-render each batch through another training subject's channel covariance operator (HISTORY.md)
    reve_scratch_optx_things)   echo "$(recipe reve_scratch_things) --op_transfer_prob 0.5" ;;
    reve_d6_optx_things)        echo "$(recipe reve_d6_things) --op_transfer_prob 0.5" ;;
    sqf_dbn_optx_things)        echo "$(recipe sqf_dbn_things) --op_transfer_prob 0.5" ;;
    tsc_dbn40_bn64_optx_things) echo "$(recipe tsc_dbn40_bn64_things) --op_transfer_prob 0.5" ;;
    rmix_reg_optx_things)       echo "$(recipe rmix_reg_things) --op_transfer_prob 0.5" ;;
    *) return 1 ;;
  esac
}
RECIPES="reve_d6_optx_things rmix_reg_optx_things sqf_dbn_optx_things tsc_dbn40_bn64_optx_things reve_scratch_optx_things
reve_d6_things rmix_reg_things sqf_dbn_things tsc_dbn40_bn64_things reve_scratch_things"

if [ "$1" = --list ]; then for r in $RECIPES; do echo "$r: $(recipe $r)"; done; exit 0; fi
NAME=$1; shift
SPEC=$(recipe "$NAME") || { echo "unknown recipe '$NAME' (cv4.sh --list)"; exit 1; }
ARM=${SPEC%% *} FLAGS="${SPEC#* }"
DATA=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench
grep -q things_eeg2_subjects $DATA/info.json || { echo "THINGS-EEG2 subjects not exported (export_things_to_alljoined.py)"; exit 1; }
# their training caches must exist on the NAS first: built once there, staged like everyone else's, instead of
# every job rebuilding ~10 GB into its node's /tmp. The marker is written by hand once they are built.
[ -f $DATA/.cache/things_caches_ok ] || { echo "THINGS-EEG2 training caches not built on the NAS yet"; exit 1; }
for K in ${@:-1 2 3 4}; do
  TEST=${FOLDS[$K]}; [ -n "$TEST" ] || { echo "no fold $K"; exit 1; }
  OUT=results/alljoined/track1/cv4/$NAME/fold$K/$ARM
  if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip $NAME fold$K (scored)"; continue; fi
  TRAIN=""; for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || TRAIN+="$s "; done
  TRAIN+="$(seq -s ' ' 21 30)"
  echo "=== $(date '+%F %T') $NAME fold$K  train: $TRAIN test: $TEST"
  # sealed-rule score of every test subject on a seeded half of its trials (~40 repeats/image), every epoch
  SPLIT=cv4/$NAME/fold$K TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh "$ARM" \
    --select_best_on fixed --sealed_test_every 1 --sealed_test_fraction 0.5 $FLAGS \
    || echo "FAILED $NAME fold$K"
done
echo "done $NAME folds ${@:-1 2 3 4}"
