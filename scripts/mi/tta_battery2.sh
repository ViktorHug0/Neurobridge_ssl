#!/bin/bash
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
while pgrep -f tta_battery.sh >/dev/null; do sleep 15; done
r () { tag=$1; shift; env "$@" TAG=_$tag ./run.sh e20_att_tta.py $ARMX; }
ARMX=tsconv/seed3300 r tsconv_intime MODE=intime LR=0.003
ARMX=atm128/seed3300 r s3300_intime MODE=intime LR=0.003
ARMX=tsconv/seed3300 r tsconv_inmix003 MODE=inmix LR=0.003
ARMX=labram/seed3300 r labram_inmix003 MODE=inmix LR=0.003
ARMX=eegnet/seed3300 r eegnet_inmix003 MODE=inmix LR=0.003
ARMX=conformer/seed3300 r conformer_inmix003 MODE=inmix LR=0.003
ARMX=tsconv_vc/seed3300 r tsconvvc_inmix003 MODE=inmix LR=0.003
