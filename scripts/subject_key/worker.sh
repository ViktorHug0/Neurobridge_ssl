#!/bin/bash
# GPU worker of the subject-key experiments (the body of the `key_gpu` Slurm job): runs the lines of queue.txt one at
# a time, first in first out. A line is `<name> | <command>`; the command runs from the repo root, log in
# logs/subject_key/<name>.log. Add lines with q.sh (same lock). Never exits: the job holds the GPU until scancel.
R=/nasbrain/p20fores/Neurobridge_SSL; Q=$R/scripts/subject_key/queue.txt; L=$R/logs/subject_key
export PATH=/usr/bin:/bin PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R; touch $Q
while true; do
  line=$(flock $Q.lock sh -c "head -n 1 $Q; sed -i 1d $Q")
  [ -z "$line" ] && { sleep 20; continue; }
  name=$(echo "${line%%|*}" | xargs); cmd=${line#*|}
  echo "$(date '+%F %T') START $name" >> $Q.done
  bash -c "$cmd" > $L/$name.log 2>&1
  echo "$(date '+%F %T') END($?) $name |$cmd" >> $Q.done
done
