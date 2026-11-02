#!/bin/bash
#SBATCH --job-name=fusion-three
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --array=2-3
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/fusion_three_20260915/G%a-%j.log
set -euo pipefail
exec bash /nasbrain/p20fores/Neurobridge_SSL/ensemble_experiments/fusion_three/run.sh "G${SLURM_ARRAY_TASK_ID}"
