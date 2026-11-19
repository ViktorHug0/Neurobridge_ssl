#!/bin/bash
# f4_read.sh: last queue line of the fold-4 confirmation. Waits for the seven fold-4 runs, caches every member's outputs on the fold-4
# held-out people (u01_f4_*), then reads sub25b's nine (C5) and the ST x2 variant (sub26) against the sub16-like six on fold 4 ALONE,
# and the foreign-cohort drop / double screen with fold-4 models. Logs: logs/mi/f4_read.out
cd /nasbrain/p20fores/Neurobridge_SSL; C=results/alljoined/track1/cv4
for r in mi_prev_ho5k_lag5_jit3_soft3 mi_plab_ho5k_lag5_jit3 mi_stf_ho5k_lag5_jit3_soft3 mi_rmix_ho5k_lag5_jit3_soft3 mi_stf_ho5k_lag5_jit3 mi_plab_ho5k_lag5_long_soft3 mi_sqf_ho5k_lag5; do
  # not ready: put this line back at the END of the list (other lines go first) and free the worker after 10 minutes
  ls $C/$r/fold4/*/*/track1_score_sealed.json >/dev/null 2>&1 || { Q=logs/mi/queue; until mkdir $Q/lock 2>/dev/null; do sleep 1; done; echo "bash $Q/f4_read.sh" >> $Q/night.list; rmdir $Q/lock; sleep 600; exit 0; }; done
{ date
PYTHONPATH=.:scripts/mi .venv/bin/python - <<'PY'
from nc_lib import *
k = 4; X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X = torch.as_tensor(X[:n])
for s in ['mi_d6_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_rmix_ho5k_lag5_jit3:checkpoint_ep10.pth', 'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth',
          'mi_scr_ho5k_lag5:checkpoint_test_best.pth', 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth', 'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth',
          'mi_plab_ho5k_lag5_long:checkpoint_ep10.pth', 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth', 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth',
          'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth', 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth']:
    r, c = s.split(':'); p = f'{OUT}/u01_f{k}_{r}_{c}.npy'
    if not os.path.exists(p):
        f = load(r, k, c); np.save(p, f(X, bs=int(os.environ.get('BS', 256))).numpy().astype(np.float16)); del f; torch.cuda.empty_cache(); print('cached', s, flush=True)
PY
C5="0:mi_d6_ho5k_lag5_jit3@-1,1:mi_rmix_ho5k_lag5_jit3@-1,1:mi_rmix_ho5k_lag5_jit3_soft3,5:mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth,0:mi_prev_ho5k_lag5_jit3_soft3"
for W in 1 2; do echo "== fold 4, sub25b's nine, ST x$W, vs the sub16-like six"; REF=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth TRIPLES=1 CS=0.02 ADD="$C5,0:mi_stf_ho5k_lag5_jit3_soft3@$W,0:mi_stf_ho5k_lag5_jit3@$W" PYTHONPATH=.:scripts/mi .venv/bin/python scripts/mi/nv01_mean_shift.py 4 2>&1 | grep "c +0.00" | cut -c1-330; done
for A in 1 1.5 2 3; do echo "== fold 4, sub25b's nine, repeat weighting POW $A"; POW=$A REF=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth TRIPLES=1 CS=0.02 ADD="$C5,0:mi_stf_ho5k_lag5_jit3_soft3,0:mi_stf_ho5k_lag5_jit3" PYTHONPATH=.:scripts/mi .venv/bin/python scripts/mi/nv01_mean_shift.py 4 2>&1 | grep "c +0.00" | cut -c1-330; done
echo "== ds007964, fold-4 models, base sub25b's nine"
DROP="d6_ho5k_lag5_jit3,mi_rmix_ho5k_lag5_jit3_checkpoint_ep10.pth,d6_ho5k_lag5_jit3_soft3" ADD="stf_ho5k_lag5_jit3,prev_ho5k_lag5_jit3_soft3" PYTHONPATH=.:scripts/mi .venv/bin/python scripts/mi/ny05_foreign_screen.py ds007964 4 2>&1 | grep -v "Warning\|montage\|Subjects\|LaBraM\]" | sed -n '1,/ADDED/p' | cut -c1-130
echo "== member weights chosen on folds 1-3, checked on fold 4"; PYTHONPATH=.:scripts/mi .venv/bin/python scripts/mi/ny10_weight_search.py 1,2,3 4 2>&1 | grep -v "Warning\|montage\|Subjects"
date; } > logs/mi/f4_read.out 2>&1
