#!/bin/bash
# 2026-10-12 night (ALIGN_LOG.md): shared alignment screen queue (folds 1-2); several workers walk it, claims dedupe
cd /nasbrain/p20fores/Neurobridge_SSL
for r in tsc_dbn40_ml_things tsc_dbn40_ravg_things tsc_dbn40_blur_things tsc_dbn40_trunc_things tsc_dbn40_gnegc_things tsc_dbn40_siglip_things tsc_dbn40_softclip_things reve_d6_ravg_things reve_d6_trunc_things reve_d6_ml_things reve_d6_blur_things tsc_dbn40_rkd_things tsc_dbn40_gsoft_things tsc_dbn40_bn64_things; do
  for k in 1 2; do bash neurips_challenge/cv4.sh $r $k; done
done
