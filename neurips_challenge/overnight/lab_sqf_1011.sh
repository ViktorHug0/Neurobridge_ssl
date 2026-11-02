#!/bin/bash
# 2026-10-11, claim-walked: confirm scratch LaBraM + THINGS on folds 3-4, then Squeezeformer + THINGS with per-dataset BatchNorm (all folds)
cd /nasbrain/p20fores/Neurobridge_SSL
bash neurips_challenge/cv4.sh lab_s10_things 3 4
bash neurips_challenge/cv4.sh sqf_dbn_things 1 2 3 4
