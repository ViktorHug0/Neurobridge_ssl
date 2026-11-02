#!/bin/bash
# rct -> REVE-6 bridge (alljoined_encoders.BridgeTransformer), cv4 folds 1-2; REVE-component configs first
cd /nasbrain/p20fores/Neurobridge_SSL
for r in br4_things br3_things br1_things br2_things; do bash neurips_challenge/cv4.sh $r 1 2; done
