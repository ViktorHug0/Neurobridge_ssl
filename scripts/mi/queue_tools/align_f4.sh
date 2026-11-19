#!/bin/bash
# align_f4.sh: the alignment's price and gains on fold 4 (fourth fold of people), once the fold-4 lag-only rmix detector is trained.
# Requeues itself at the end of the list while the detector is not ready. Output: logs/mi/s09_f4.out
cd /nasbrain/p20fores/Neurobridge_SSL; Q=logs/mi/queue
ls results/alljoined/track1/cv4/mi_rmix_ho5k_lag5/fold4/*/*/track1_score_sealed.json >/dev/null 2>&1 || { until mkdir $Q/lock 2>/dev/null; do sleep 1; done; echo "bash $Q/align_f4.sh" >> $Q/night.list; rmdir $Q/lock; sleep 600; exit 0; }
export PYTHONPATH=.:scripts/mi; P=.venv/bin/python
{ $P scripts/mi/nr05_autolag.py mi_rmix_ho5k_lag5 4 checkpoint_ep10.pth 2>&1 | grep -v "Warning\|montage\|Subjects" | tail -3
  $P scripts/mi/nr07_autolag_e2e.py 4 2>&1 | grep "offset\|parabola, step 2, +1.5\|zero"
  $P scripts/mi/ns09_align_policy.py 1,2,3,4 2>&1 | grep -v "Warning\|montage\|Subjects"
} > logs/mi/s09_f4.out 2>&1
