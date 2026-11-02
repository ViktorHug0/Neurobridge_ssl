#!/bin/bash
# REVE dissection -> TSConv (cv4 folds 1-2): input z-score, and per-dataset layers for THINGS. Claim-walked.
cd /nasbrain/p20fores/Neurobridge_SSL
for r in tsc_dspat_things tsc_e20_things tsc_dbn_things tsc_inorm; do bash neurips_challenge/cv4.sh $r 1 2; done
