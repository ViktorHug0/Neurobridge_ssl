#!/bin/bash
# 2026-10-10, cv4 folds 1-2, claim-walked: scratch LaBraM + THINGS (longest first), mixup on scratch REVE-6 + THINGS,
# then the 6-block "small LaBraM" + THINGS
cd /nasbrain/p20fores/Neurobridge_SSL
for r in lab_s10_things reve_d6_mix_things lab_d6_things; do bash neurips_challenge/cv4.sh $r 1 2; done
