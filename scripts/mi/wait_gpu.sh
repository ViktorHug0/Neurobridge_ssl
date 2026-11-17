#!/bin/bash
# wait_gpu.sh <slurm job id | local>: block until that GPU has had no srun step (local: no GPU process) on 3 checks in a row, one minute apart
n=0
while [ $n -lt 3 ]; do
  sleep 60
  if [ "$1" = local ]; then busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); else busy=$(squeue -h -s -j $1 -o %i | grep -vcE "batch|extern"); fi
  [ "$busy" = 0 ] && n=$((n+1)) || n=0
done
