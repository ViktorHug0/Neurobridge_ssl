#!/bin/bash
# One CPU-only Slurm job (user grant, 2026-11-10): runs the command file given as argument; on 2026-11-10 the appendix
# grid (inc/v3/full: menu size x repetitions x stream, all embedding-level methods) and the model-level baselines.
#   sbatch --job-name inccpu --account brain --qos low --partition Brain_GPU -c 24 --mem 58G --time 12:00:00 \
#     --output logs/cvpr/inc/slurm_%j.out CVPR_paper/scripts/inc_slurm.sh
cd /nasbrain/p20fores/Neurobridge_SSL
# batch jobs have no Kerberos ticket for the NFS home: keep it out of PATH/HOME (as gpu_worker.sh)
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home
mkdir -p "$HOME"
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/nasbrain/p20fores/SATTC-CVPR2026/.deps
xargs -a ${1:?file with one command per line, e.g. the last lines of inc_queue.sh} -P ${2:-22} -d '\n' -I{} bash -c '{}'
