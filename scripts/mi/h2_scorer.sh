#!/bin/bash
# every 10 min for 15 h: score each finished second-proxy run against its reference once (nq09_proxy2.py) -> logs/mi/h2_scores.txt
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi; O=../../logs/mi/h2_scores.txt
fin() { grep -aq track1_score_sealed.json ../../logs/mi/train_$1_f$2.log 2>/dev/null; }
for i in $(seq 90); do
  for k in 1 3; do for pair in mi_rmix_h2_lag5:mi_rmix_h2_lag5_ds2 mi_rmix_h2_lag5:mi_rmix_h2_lag5_bank mi_rmix_h2_lag5:mi_rmix_h2_lag5_ds1 mi_rmix_h2_lag5:mi_rmix_h2_lag5_ds4 mi_d6_h2_lag5:mi_d6_h2_lag5_ds2 mi_sqfx_h2_lag5:mi_sqfx_h2_lag5_ds2; do
    r=${pair%%:*} c=${pair##*:}; [[ $r == *sqfx* ]] && E="40 32 36" || E="10 8 12"
    if fin $r $k && fin $c $k && ! grep -q "^## $c f$k " $O 2>/dev/null; then
      echo "## $c f$k $(date +%H:%M)" >> $O
      ../../.venv/bin/python nq09_proxy2.py $k $(for e in $E; do echo $r:$e; done) $(for e in $E; do echo $c:$e; done) 2>&1 | grep -E "@|Error" >> $O
    fi
  done; done
  for k in 1 3; do for c in mi_rmix_ho5k_lag5_g18 mi_rmix_ho5k_lag5_g18bank mi_rmix_ho5k_lag5_rm50 mi_rmix_ho5k_lag5_rm100 mi_d6_ho5k_lag5_rm50; do  # first-proxy runs (ds004018): alone and swapped into the lag five
    if fin $c $k && ! grep -q "^## $c f$k " $O 2>/dev/null; then
      echo "## $c f$k $(date +%H:%M)" >> $O; ../../.venv/bin/python ni01_new_member.py $c $k 8,10,12 2>&1 | grep -E "alone|swapped for (rmix|d6)" | grep -E "5k r3|5k r4|14k r3" >> $O
    fi
  done; done
  sleep 600
done
