#!/bin/bash
# CPU worker: same as gpu_worker.sh but pops CVPR_paper/queue/cpu_queue.txt. Launch WITHOUT `srun -c` (with -c, all
# --overlap steps get pinned to the same few cores: 13 analyses once shared 4 CPUs). Threads per item = $THREADS.
#   srun --jobid=<job> --overlap -n 1 env THREADS=3 bash CVPR_paper/scripts/cpu_worker.sh
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth
export OMP_NUM_THREADS=${THREADS:-3} OPENBLAS_NUM_THREADS=${THREADS:-3} MKL_NUM_THREADS=${THREADS:-3}
mkdir -p "$HOME"
Q=CVPR_paper/queue/cpu_queue.txt
while true; do
  line=$(flock "$Q.lock" /bin/bash -c "l=\$(grep -v -m1 -e '^#' -e '^\s*$' '$Q'); [ -n \"\$l\" ] && grep -vxF -- \"\$l\" '$Q' > '$Q.tmp'; [ -n \"\$l\" ] && cat '$Q.tmp' > '$Q'; echo \"\$l\"")
  # POLL=1: wait for new items instead of exiting (a batch job ends, killing all its steps, once its workers exit)
  [ -z "$line" ] && [ "${POLL:-0}" = 1 ] && { sleep 120; continue; }
  [ -z "$line" ] && { echo "[$(date '+%F %T')] queue empty, exiting"; return 0; }
  echo "$(date '+%F %T') $(hostname) $line" >> CVPR_paper/queue/cpu_started.txt
  /bin/bash -c "$line"; rc=$?
  echo "$(date '+%F %T') $(hostname) rc=$rc $line" >> CVPR_paper/queue/cpu_done.txt
done
}
main "$@"; exit $?
