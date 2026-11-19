#!/bin/bash
# y08_job.sh replay | <fold>: queue lines for the stateless-correction test (ny08) and the replay of the hardened package (test_hard)
cd /nasbrain/p20fores/Neurobridge_SSL
if [ "$1" = replay ]; then
  for d in 0 4 -4; do .venv/bin/python neurips_challenge/package_submission.py neurips_challenge/submissions/test_hard --verify 1 --delay $d 2>&1 | grep "fold 1\|Error\|left out" | sed "s/^/delay $d: /"; done > logs/mi/test_hard_replay.out
else
  PYTHONPATH=.:scripts/mi .venv/bin/python scripts/mi/ny08_stateless_input.py $1 2>&1 | grep -v "Warning\|montage\|Subjects\|LaBraM\]" > logs/mi/y08_f$1.out
fi
