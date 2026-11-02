#!/bin/bash
# REVE scaling curves on fold 2 (fold 1 done 2026-10-07), then the widened robust transformer screen.
cd /nasbrain/p20fores/Neurobridge_SSL
export REVE_BOTH_FOLDS=1
bash neurips_challenge/scaling.sh reve 2
THINGS=last bash neurips_challenge/scaling.sh reve 2
THINGS=first bash neurips_challenge/scaling.sh reve 2
bash neurips_challenge/cv4.sh rct_w512_things 1 2
