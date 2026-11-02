#!/bin/bash
# GPU worker: pops the first non-comment line of CVPR_paper/queue/gpu_queue.txt (under flock) and runs it, until the
# queue is empty, then exits (releasing the GPU). Several workers can share the queue. Edit the queue freely; a
# line is gone from it once a worker has taken it (see started.txt / the worker log).
#   sbatch --job-name cvprq --account brain --qos low --partition Brain_GPU --gres gpu:rtx3080:1 -c 7 --mem 28G \
#     --time 2-00:00:00 --output logs/cvpr/worker_%j.out CVPR_paper/scripts/gpu_worker.sh
cd /nasbrain/p20fores/Neurobridge_SSL
# batch jobs have no Kerberos ticket for the NFS home: keep it out of PATH/HOME (else exec fails with EKEYEXPIRED)
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface
mkdir -p "$HOME"
Q=CVPR_paper/queue/gpu_queue.txt
while true; do
  line=$(flock "$Q.lock" /bin/bash -c "l=\$(grep -v -m1 -e '^#' -e '^\s*$' '$Q'); [ -n \"\$l\" ] && grep -vxF -- \"\$l\" '$Q' > '$Q.tmp'; [ -n \"\$l\" ] && cat '$Q.tmp' > '$Q'; echo \"\$l\"")
  [ -z "$line" ] && { echo "[$(date '+%F %T')] queue empty, exiting"; exit 0; }
  echo "$(date '+%F %T') ${SLURM_JOB_ID} $line" >> CVPR_paper/queue/started.txt
  echo "[$(date '+%F %T')] START $line"
  /bin/bash -c "$line"; rc=$?
  echo "[$(date '+%F %T')] EXIT $rc $line"
  [ $rc -ne 0 ] && echo "$(date '+%F %T') ${SLURM_JOB_ID} rc=$rc $line" >> CVPR_paper/queue/failed.txt
done
