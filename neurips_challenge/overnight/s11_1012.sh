#!/bin/bash
# 2026-10-12 ROUTING_LOG E27: 11-test-subject testbed for the batch transform (ensemble recipes, 2 splits)
cd /nasbrain/p20fores/Neurobridge_SSL
for r in reve_d6_things sqf_dbn_things tsc_dbn40_things; do for k in 1 2; do bash neurips_challenge/s11.sh $r $k; done; done
