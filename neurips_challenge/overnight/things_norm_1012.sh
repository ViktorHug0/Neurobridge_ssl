#!/bin/bash
# 2026-10-12 ROUTING_LOG phase 2: THINGS input normalisations, screen on folds 1-2
cd /nasbrain/p20fores/Neurobridge_SSL
for r in reve_d6_coral_things reve_d6_car_things sqf_dbn_coral_things; do bash neurips_challenge/cv4.sh $r 1 2; done
