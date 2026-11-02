#!/bin/bash
# leave-one-out split of REVE's backbone advantage (bridge B2 -> B3), cv4 folds 1-2
cd /nasbrain/p20fores/Neurobridge_SSL
for r in sw0_things sw1_things sw2_things sw3_things sw4_things sw5_things; do bash neurips_challenge/cv4.sh $r 1 2; done
