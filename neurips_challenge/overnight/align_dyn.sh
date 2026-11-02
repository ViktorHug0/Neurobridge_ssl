#!/bin/bash
# 2026-10-12 night (ALIGN_LOG.md): editable alignment queue. Re-reads align_queue.txt ("<recipe> <fold>" per line)
# before every run, so the queue can be reordered while workers run; exits when nothing is left to run.
# Skips a run when it is scored, or claimed by ANOTHER job that Slurm still reports alive (dead claims are taken over).
cd /nasbrain/p20fores/Neurobridge_SSL
Q=${Q:-neurips_challenge/overnight/align_queue.txt}  # Q=...: another queue file (aug_queue.txt)
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home; mkdir -p $HOME  # off the Kerberos /homes (common.sh)
ME=${SLURM_JOB_ID:-$$}
alive() { [[ "$(sacct -n -X -j "$1" -o State 2>/dev/null | head -1 | tr -d ' ')" =~ ^(RUNNING|PENDING|REQUEUED|SUSPENDED|)$ ]]; }
while true; do
  ran=0
  while read -r r k; do
    [ -z "$r" ] || [[ $r == \#* ]] && continue
    out=$(ls -d results/alljoined/track1/cv4/$r/fold$k/*/ 2>/dev/null | head -1)
    if [ -n "$out" ]; then
      ls ${out}2026*/track1_score.json >/dev/null 2>&1 && continue
      owner=$(cat ${out}.claim/job 2>/dev/null)
      [ -n "$owner" ] && [ "$owner" != "$ME" ] && alive "$owner" && continue
    fi
    bash neurips_challenge/cv4.sh $r $k; ran=1; break
  done < $Q
  [ $ran = 0 ] && break
done
echo "align_dyn: queue empty"
