#!/bin/bash
#SBATCH --job-name=f1-replay-finish
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-tp-br-513
#SBATCH --cpus-per-task=6
#SBATCH --mem=29G
#SBATCH --time=12:00:00
#SBATCH --array=4-5
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/f1_checkpoint_replay_20260915/relaunch-sub-%a-%j.log
set -euo pipefail
exec bash /nasbrain/p20fores/Neurobridge_SSL/ensemble_experiments/f1_replay_snapshot/run.sh "$SLURM_ARRAY_TASK_ID"
