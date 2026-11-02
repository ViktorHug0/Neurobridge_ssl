#!/bin/bash
# v2 paper (2026-10-24): queues the CPU evaluation of each ValCon arm once its GPU folds exist, so no CPU worker sits
# in a wait loop. One line of PENDING = "<loso or intra dir> <n folds> <cpu queue command>"; done items go to
# CVPR_paper/queue/vc_driver_done.txt. Run with nohup on the interactive node; it exits when nothing is pending.
cd /nasbrain/p20fores/Neurobridge_SSL
D=CVPR_paper/queue/vc_driver_done.txt
touch $D
PENDING=()
# 2026-10-25: ATM-128 image-target variants for the RSA analysis (Fig. 2b), R1 battery as for the encoders
for v in iv20 iv44 dino40 cliph bigg27; do
  PENDING+=("results/things_eeg/cvpr/loso/atm128_${v}_vc/seed3300 10 bash CVPR_paper/scripts/encoder_battery.sh atm128_${v}_vc 3300 > logs/cvpr/final/battery_atm128_${v}_vc.log 2>&1")
done
# ATM training variants dropped from the paper (user, 2026-10-24): no screen3 items any more
while true; do
  left=0
  for item in "${PENDING[@]}"; do
    read -r dir n cmd <<< "$item"
    grep -qxF -- "$cmd" $D && continue
    left=$((left + 1))
    if [ $(ls $dir/*-sub-*/result.csv 2>/dev/null | wc -l) -ge $n ]; then
      flock CVPR_paper/queue/cpu_queue.txt.lock /bin/bash -c "echo '$cmd' >> CVPR_paper/queue/cpu_queue.txt"
      echo "$cmd" >> $D
      echo "[$(date '+%F %T')] queued: $cmd"
    fi
  done
  [ $left -eq 0 ] && { echo "[$(date '+%F %T')] all queued"; exit 0; }
  sleep 600
done
