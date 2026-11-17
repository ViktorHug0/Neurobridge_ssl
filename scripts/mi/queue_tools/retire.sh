#!/bin/bash
# retire.sh: the first worker that reaches this queue line stops taking work (user, 16 Nov 20:25: "we are on a new interactive node; as soon
# as one of the 4 GPUs releases, stop using it and switch to this node"). Its job goes to logs/mi/queue/retired; wait519.sh on the new
# interactive node then starts a worker there, so four GPUs are in use at every moment. The holder job itself stays (its `sleep 2d`).
cd /nasbrain/p20fores/Neurobridge_SSL
echo "$SLURM_JOB_ID $(hostname) $(date '+%d %H:%M')" > logs/mi/queue/retired
echo "$(date '+%d %H:%M') RETIRED: job $SLURM_JOB_ID on $(hostname) takes no more work" >> logs/mi/queue/night.log
kill $PPID
