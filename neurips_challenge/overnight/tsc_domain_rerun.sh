#!/bin/bash
# rerun after the unknown-subject routing fix (scored subjects had taken the THINGS path), cv4 folds 1-2, claim-walked
cd /nasbrain/p20fores/Neurobridge_SSL
for r in tsc_dspat_things tsc_dbn_things; do bash neurips_challenge/cv4.sh $r 1 2; done
