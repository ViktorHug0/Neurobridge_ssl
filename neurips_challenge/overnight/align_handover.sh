#!/bin/bash
# hand each GPU from the stale-list workers to align_dyn.sh as soon as their current run is scored
cd /nasbrain/p20fores/Neurobridge_SSL
scored() { ls results/alljoined/track1/cv4/$1/fold$2/*/2026*/track1_score.json >/dev/null 2>&1; }
tree() { local p; for p in $(ps -o pid= --ppid $1); do tree $p; done; echo $1; }
done1=0; done2=0; done3=0
while [ $((done1 + done2 + done3)) -lt 3 ]; do
  if [ $done1 = 0 ] && scored tsc_dbn40_bn64_things 1; then scancel 60326; done1=1; echo "$(date +%T) handed 513 (60326)"; fi
  if [ $done2 = 0 ] && scored tsc_dbn40_bn64_things 2; then scancel 60327; done2=1; echo "$(date +%T) handed 514 (60327)"; fi
  if [ $done3 = 0 ] && scored tsc_dbn40_ravg_things 1; then
    q=$(pgrep -xf "bash neurips_challenge/overnight/align_queue_1012.sh" | head -1)
    [ -n "$q" ] && kill $(tree $q); done3=1; echo "$(date +%T) handed 516 (killed stale list $q)"
  fi
  sleep 20
done
