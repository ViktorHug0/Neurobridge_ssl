#!/bin/bash
#SBATCH --job-name=sage_rank
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/nsd_sage/rank_perpool_%j.log

# The interactive session runs under a 1925M cgroup, which SIGKILLs this job during load_full
# (two 1.6 GB tensors per subject). Run it here instead.
cd /nasbrain/p20fores/Neurobridge_SSL
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
srun .venv/bin/python -u scripts/nsd_sage/per_pool_sage.py --sweep --ks 384 576 596 598
