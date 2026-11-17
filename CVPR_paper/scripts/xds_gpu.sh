#!/bin/bash
# One GPU Slurm job (user grant, 2026-11-15: one GPU for the Alljoined / THINGS-MEG trainings): runs the commands of
# the file given as argument one after the other.
#   sbatch -p Brain_GPU -J cvpr_xds --gres=gpu:rtx3080:1 -c 8 --mem=30G -t 1-00:00:00 -x sl-mee-br-302 \
#     -o logs/cvpr/xds/slurm_%j.out CVPR_paper/scripts/xds_gpu.sh <queue file>
cd /nasbrain/p20fores/Neurobridge_SSL
while read -r line; do bash -c "$line"; done < "${1:?queue file}"
