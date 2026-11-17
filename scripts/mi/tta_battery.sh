#!/bin/bash
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
r () { tag=$1; shift; env "$@" TAG=_$tag ./run.sh e20_att_tta.py $ARMX; }
ARMX=atm128_vc/seed3300 r vc3300_att MODE=att
ARMX=atm128/seed3300 r s3300_inmix01 MODE=inmix LR=0.01
ARMX=atm128/seed3300 r s3300_inmix003 MODE=inmix LR=0.003
ARMX=tsconv/seed3300 r tsconv_inmix01 MODE=inmix LR=0.01
ARMX=tsconv/seed3300 r tsconv_out MODE=out LR=0.001
ARMX=atm128_vc/seed3301 r vc3301_att MODE=att
ARMX=atm128_vc/seed3302 r vc3302_att MODE=att
ARMX=atm128/seed3300 r s3300_att_gala MODE=att HEAD=gala PARAMS=results/things_eeg/cvpr/tta_screen_loso/atm128_s3300/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
ARMX=sqf/seed3300 r sqf_inmix01 MODE=inmix LR=0.01
ARMX=eegproject/seed3300 r ep_inmix01 MODE=inmix LR=0.01
ARMX=atm128/seed3301 r s3301_att MODE=att
ARMX=atm128/seed3302 r s3302_att MODE=att
