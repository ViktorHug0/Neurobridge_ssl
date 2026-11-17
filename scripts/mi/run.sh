#!/bin/bash
# run a script as a step inside the mi_gpu allocation: ./run.sh e00_check.py [args]; log in logs/mi/<name>.log
J=$(squeue -u $USER -h -n mi_gpu -o %i | head -1)
mkdir -p /nasbrain/p20fores/Neurobridge_SSL/logs/mi
n=$(basename $1 .py)${TAG:+_$TAG}
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/mi
srun --jobid=$J --overlap /usr/bin/env PATH=/usr/bin:/bin OMP_NUM_THREADS=${OMP:-8} /nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python -u "$@" > /nasbrain/p20fores/Neurobridge_SSL/logs/mi/$n.log 2>&1
