#!/bin/bash
# 2026-10-09 night, cv4 folds 1-2: final-norm transfer runs, then the fine-tune ladder from the core probe.
# Every walker runs this same list; cv4 claims split the work. Fine-tunes need their fold's probe scored, so
# the ladder is walked twice: the second pass picks up fine-tunes whose probe another walker finished meanwhile.
cd /nasbrain/p20fores/Neurobridge_SSL
C=results/alljoined/track1/cv4
# the probe's fold-2 claim was parked on interactive job 59442 to keep the old walkers off it
[ "$(cat $C/reve_lp_things/fold2/reve/.claim/job 2>/dev/null)" = 59442 ] && rm -rf $C/reve_lp_things/fold2/reve/.claim
for r in rct_nofn_things lab_s10_nofn plab_gain_things_nofn; do bash neurips_challenge/cv4.sh $r 1 2; done
ladder() {
  for K in 1 2; do
    bash neurips_challenge/cv4.sh reve_lp_things $K
    run=$(ls -d $C/reve_lp_things/fold$K/reve/2026* 2>/dev/null | tail -1)
    [ -f "$run/track1_score.json" ] || { echo "probe fold$K not scored yet: its fine-tunes wait for the second pass"; continue; }
    for F in reve_ft_things reve_ftmix_things reve_ftours_things; do
      CV4_EXTRA="--init_checkpoint $run/checkpoint_sealed_best.pth" bash neurips_challenge/cv4.sh $F $K
    done
  done
}
ladder
for r in sw0_things sw5_things; do bash neurips_challenge/cv4.sh $r 3 4; done  # filler: confirm the final-norm loss on folds 3-4
ladder
