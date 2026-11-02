#!/bin/bash
# REVE authors' two-stage fine-tuning on cv4 folds: probe, then fine-tune from the probe's best-epoch checkpoint.
# Two variants: lp/ft = the two-stage core with our optimizer setup; lpa/fta = plus the rest of the authors' recipe.
# The worker that runs a fold's probe runs its fine-tune right after, so a fine-tune never starts from an unfinished probe.
cd /nasbrain/p20fores/Neurobridge_SSL
for K in ${@:-1 2 3 4}; do
  for V in "" a; do
    bash neurips_challenge/cv4.sh reve_lp${V}_things $K
    run=$(ls -d results/alljoined/track1/cv4/reve_lp${V}_things/fold$K/reve/2026* 2>/dev/null | tail -1)
    [ -f "$run/track1_score.json" ] || { echo "probe lp$V fold$K not finished here: skip its fine-tune"; continue; }
    CV4_EXTRA="--init_checkpoint $run/checkpoint_sealed_best.pth" bash neurips_challenge/cv4.sh reve_ft${V}_things $K
  done
done
