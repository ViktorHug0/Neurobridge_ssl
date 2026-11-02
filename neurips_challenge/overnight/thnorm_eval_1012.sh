#!/bin/bash
# 2026-10-12 ROUTING_LOG E44: after the THINGS-normalisation screen (jobs 60064/60065), cache trial outputs and compare
# each recipe with its baseline on folds 1-2, plain and under the deployed router. Runs as steps of the interactive
# allocation (CPU jobs stay on the user's 3 allowed nodes).
cd /nasbrain/p20fores/Neurobridge_SSL
while squeue -j 60064,60065 -h 2>/dev/null | grep -q .; do sleep 60; done
S="srun --jobid=59605 --overlap -n 1 -c 6 --mem=24G"
for r in reve_d6_coral_things reve_d6_car_things sqf_dbn_coral_things; do $S .venv/bin/python neurips_challenge/trial_cache.py --device cpu $r; done
.venv/bin/python neurips_challenge/cv4_summary.py --ref reve_d6_things 2>&1 | grep -E "reve_d6_(coral_|car_)?things"
.venv/bin/python neurips_challenge/cv4_summary.py --ref sqf_dbn_things 2>&1 | grep -E "sqf_dbn_(coral_)?things"
for r in reve_d6_things reve_d6_coral_things reve_d6_car_things sqf_dbn_things sqf_dbn_coral_things; do
  echo "##### $r"; $S .venv/bin/python neurips_challenge/n11.py --cv --cfgs deploy $r 2>&1 | grep -vE "Warning|^draw|per subject"
done
