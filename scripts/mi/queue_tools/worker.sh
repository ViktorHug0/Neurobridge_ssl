#!/bin/bash
# worker.sh <tag>: runs ON the GPU's node (as one srun step of a holder job, or directly on the interactive node): takes the next line
# of logs/mi/queue/night.list (a full command) until the list is empty. The list, the log and this script live on the NAS so that a
# change of interactive node loses nothing (15-16 Nov: the queue in the node-local scratchpad was lost with node 516).
cd /nasbrain/p20fores/Neurobridge_SSL; Q=logs/mi/queue
until mkdir $Q/lock 2>/dev/null; do sleep 1; done  # lines parked by fit.sh go back to the head of the list
[ -s $Q/night.deferred ] && { cat $Q/night.deferred $Q/night.list > $Q/nl.tmp; command mv -f $Q/nl.tmp $Q/night.list; : > $Q/night.deferred; }; rmdir $Q/lock
while :; do
  until mkdir $Q/lock 2>/dev/null; do sleep 1; done  # mkdir is atomic on NFS
  line=$(head -1 $Q/night.list); sed -i 1d $Q/night.list; rmdir $Q/lock
  [ -z "$line" ] && break
  echo "$(date '+%d %H:%M') $1: $line" >> $Q/night.log
  $line
done
