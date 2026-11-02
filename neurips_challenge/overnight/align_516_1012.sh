#!/bin/bash
# 2026-10-12 night (ALIGN_LOG.md): alignment screens on the interactive node's GPU (one of the user's 3)
cd /nasbrain/p20fores/Neurobridge_SSL
for r in reve_d6_gneg_things reve_d6_bn64_things; do bash neurips_challenge/cv4.sh $r 1 2; done
bash neurips_challenge/overnight/align_queue_1012.sh
bash neurips_challenge/overnight/align_dyn.sh
