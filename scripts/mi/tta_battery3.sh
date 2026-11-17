#!/bin/bash
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
r () { tag=$1; shift; env "$@" TAG=_$tag ./run.sh e20_att_tta.py $ARMX; }
ARMX=atm128/seed3300 r s3300_attmix MODE=att+inmix
ARMX=atm128/seed3300 r s3300_att_plainpl MODE=att PL=plain
ARMX=tsconv/seed3300 r tsconv_inmix_plainpl MODE=inmix LR=0.003 PL=plain
ARMX=atm128/seed3300 r s3300_att_n100 MODE=att NQ=100
ARMX=atm128/seed3300 r s3300_att_n50 MODE=att NQ=50
ARMX=tsconv/seed3300 r tsconv_inmix_n100 MODE=inmix LR=0.003 NQ=100
ARMX=tsconv/seed3300 r tsconv_inmix_n50 MODE=inmix LR=0.003 NQ=50
ARMX=atm128_vc/seed3300 r vc3300_att24 MODE=att ROUNDS=24
