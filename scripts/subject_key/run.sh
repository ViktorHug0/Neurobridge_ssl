#!/bin/bash
# run a script right now as a step inside the key_gpu job, next to whatever the worker runs (small evaluations only):
#   run.sh k01_teacher.py [args]   -> logs/subject_key/<name>[_$TAG].log, and its last ${TAIL:-40} lines on stdout
# (the output comes back through srun, not through NFS: the interactive node's view of fresh files can lag by minutes)
R=/nasbrain/p20fores/Neurobridge_SSL
J=$(squeue -u $USER -h -n key_gpu -t R -o %i | head -1)
n=$(basename $1 .py)${TAG:+_$TAG}
cd $R/scripts/subject_key
srun --jobid=$J --overlap /usr/bin/env PATH=/usr/bin:/bin OMP_NUM_THREADS=4 /bin/bash -c "$R/.venv/bin/python -u $* > $R/logs/subject_key/$n.log 2>&1; grep -v 'Warning\|Subjects:\|LaBraM\|montage' $R/logs/subject_key/$n.log | tail -n ${TAIL:-40}"
