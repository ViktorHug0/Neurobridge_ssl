#!/bin/bash
# Alljoined-1.6M and THINGS-MEG under the final method (2026-11-16): waits for the stream-based selection
# (select_stream.py, results/<dataset>/cvpr/select), merges it, then runs SCORE and GALA causally with the selected
# settings -> results/<dataset>/cvpr/causal2 (zero-shot, CSLS, SATTC, Hungarian are in causal/, same streams).
# One line of an inc_slurm.sh queue; several jobs may read it, the first one takes it.
cd /nasbrain/p20fores/Neurobridge_SSL
mkdir results/alljoined/cvpr/chain.lock 2>/dev/null || exit 0
until [ "$(ls results/alljoined/cvpr/select/s3300_*_c*.csv 2>/dev/null | wc -l)" -ge 120 ] && [ "$(ls results/things_meg/cvpr/select/s3300_*_c*.csv 2>/dev/null | wc -l)" -ge 24 ]; do sleep 60; done
sleep 30
for D in alljoined things_meg; do
  .venv/bin/python CVPR_paper/scripts/select_stream.py merge --out_dir results/$D/cvpr/select || exit 1
done
.venv/bin/python - > results/alljoined/cvpr/queue_causal2.txt <<'PY'
import pandas as pd
for D, n, opt in (('alljoined', 20, '--reps 20 40 --orders 3'), ('things_meg', 4, '--reps 6 4 2')):
    R = f'results/{D}/cvpr'
    sc = pd.read_csv(f'{R}/select/score_selected_s3300.csv').set_index('subject')
    for s in range(1, n + 1):
        print(f'.venv/bin/python CVPR_paper/scripts/inc_causal.py --cache_dir {R}/cache/atm128_vc_s3300 --params_csv {R}/select/selected_per_subject_s3300.csv '
              f'--output_csv {R}/causal2/atm128_vc_s3300_{s}.csv --subjects {s} --score {int(sc.k[s])} {sc.rho[s]} --K 200 {opt} --methods score gala_sw_freq1 '
              f'> logs/cvpr/inc/xds_causal2_{D}_{s}.log 2>&1')
PY
xargs -a results/alljoined/cvpr/queue_causal2.txt -P 24 -d '\n' -I{} bash -c '{}'
