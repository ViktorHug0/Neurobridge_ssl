#!/bin/bash
# Six intervention arms split into two five-fold workers each. Four workers run
# concurrently, keeping four GPUs occupied without creating a long two-arm tail.
#SBATCH --job-name=decor_twins
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-decor-twins-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-decor-twins-%A_%a.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-8}
export DEVICE=cuda:0

arm_index=$((SLURM_ARRAY_TASK_ID / 2))
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then
  export SUBJECTS="1 2 3 4 5"
else
  export SUBJECTS="6 7 8 9 10"
fi

echo "[worker] task=$SLURM_ARRAY_TASK_ID arm=$arm_index subjects=$SUBJECTS"
bash ensemble_experiments/decorrelated_models/run_first_wave.sh "$arm_index"
