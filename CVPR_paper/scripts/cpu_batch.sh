#!/bin/bash
# Batch job: $WORKERS POLLING cpu_worker.sh (they never exit, so the job lives to its time limit: 2026-10-21 14:40
# an emptied queue ended job 63766 and killed every step inside it), plus, with GPU=1, one gpu_worker.sh.
#   sbatch --job-name cvprcpu --account brain --qos low --partition Brain_GPU --gres gpu:rtx3090:1 -c 28 --mem 56G \
#     --time 12:00:00 --output logs/cvpr/cpubatch_%j.out --export=ALL,WORKERS=10 CVPR_paper/scripts/cpu_batch.sh
cd /nasbrain/p20fores/Neurobridge_SSL
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home
[ "${GPU:-0}" = 1 ] && bash CVPR_paper/scripts/gpu_worker.sh > logs/cvpr/worker_${SLURM_JOB_ID}_gpu.out 2>&1 &
export CUDA_VISIBLE_DEVICES=
for i in $(seq ${WORKERS:-10}); do POLL=1 THREADS=2 bash CVPR_paper/scripts/cpu_worker.sh & sleep 1; done
wait
