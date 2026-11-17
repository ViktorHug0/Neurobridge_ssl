#!/bin/bash
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
P=results/things_eeg/cvpr/tta_screen_loso
r () { tag=$1; arm=$2; shift 2; env "$@" TAG=_$tag ./run.sh e20_att_tta.py $arm; }
G="HEAD=gala PARAMS=$P/atm128_vc_s3300/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv"
# cross-fold selection grids (16 rounds each)
for lr in 0.01 0.02 0.1; do r cf_att_gala_lr$lr atm128_vc/seed3300 $G MODE=att LR=$lr ROUNDS=16; done
for lr in 0.001 0.003 0.01 0.03; do r cf_tsc_inmix_lr$lr tsconv_vc/seed3300 MODE=inmix LR=$lr ROUNDS=16; done
for lr in 0.01 0.02 0.05 0.1; do r cf_att_sattc_lr$lr atm128_vc/seed3300 MODE=att LR=$lr ROUNDS=16; done
# open gallery (R3-like): 200 test images + distractors, CSLS head and full_sattc head
for nd in 200 1000; do
  r og_att_csls_d$nd atm128_vc/seed3300 MODE=att HEADNAME=csls NDIS=$nd
  r og_att_sattc_d$nd atm128_vc/seed3300 MODE=att HEADNAME=full_sattc NDIS=$nd
  r og_tsc_csls_d$nd tsconv_vc/seed3300 MODE=inmix LR=0.003 HEADNAME=csls NDIS=$nd
done
r og_att_csls_d0 atm128_vc/seed3300 MODE=att HEADNAME=csls
r og_tsc_csls_d0 tsconv_vc/seed3300 MODE=inmix LR=0.003 HEADNAME=csls
