#!/bin/bash
# fit.sh <hours> <command...>: run the command only if this Slurm job has that many hours left before its time limit; otherwise park the
# line in night.deferred and wait 10 minutes (so that a holder job near its 2-day limit does not start a run it cannot finish)
cd /nasbrain/p20fores/Neurobridge_SSL; Q=logs/mi/queue; H=$1; shift
left=$(squeue -h -j $SLURM_JOB_ID -o %L | awk -F'[-:]' '{ if ($0 ~ /[A-Z]/) print 9999; else if ($0 ~ /-/) print $1*24+$2+$3/60; else if (NF==3) print $1+$2/60; else print $1/60 }')
if awk "BEGIN{exit !(${left:-0} >= $H)}"; then exec "$@"; fi
echo "bash $Q/fit.sh $H $*" >> $Q/night.deferred  # parked: the next worker that STARTS (a replacement holder job) takes these first
echo "$(date '+%d %H:%M') deferred (${left} h left < $H): $*" >> $Q/night.log; sleep 600
