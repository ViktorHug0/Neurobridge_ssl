#!/bin/bash
# GALA head (CVPR sage_plus, each subject's cross-fold parameters) + input-side self-training, paper protocol arms
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
while pgrep -f tta_battery3.sh >/dev/null; do sleep 15; done
P=results/things_eeg/cvpr/tta_screen_loso
r () { tag=$1; arm=$2; pdir=$3; shift 3; env "$@" HEAD=gala PARAMS=$P/$pdir/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv TAG=_gala_$tag ./run.sh e20_att_tta.py $arm; }
r vc3300_att atm128_vc/seed3300 atm128_vc_s3300 MODE=att ROUNDS=16
r vc3301_att atm128_vc/seed3301 atm128_vc_s3301 MODE=att ROUNDS=16
r vc3302_att atm128_vc/seed3302 atm128_vc_s3302 MODE=att ROUNDS=16
r tsconvvc_inmix tsconv_vc/seed3300 tsconv_vc_s3300 MODE=inmix LR=0.003
r tsconv_inmix tsconv/seed3300 tsconv_s3300 MODE=inmix LR=0.003
r sqf_inmix sqf/seed3300 sqf_s3300 MODE=inmix LR=0.003
r vc3300_inmix atm128_vc/seed3300 atm128_vc_s3300 MODE=inmix LR=0.003
r s3301_att atm128/seed3301 atm128_s3301 MODE=att
r s3302_att atm128/seed3302 atm128_s3302 MODE=att
