#!/bin/bash
# Two matched five-fold workers per arm; pair each subject on the same GPU class.
#SBATCH --job-name=stoch_twins
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-stoch-twins-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-stoch-twins-%A_%a.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-8}
export DEVICE=cuda:0

if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then
  export SUBJECTS="1 2 3 4 5"
else
  export SUBJECTS="6 7 8 9 10"
fi
if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
  variant=control
else
  variant=stochastic
fi

echo "[worker] task=$SLURM_ARRAY_TASK_ID variant=$variant subjects=$SUBJECTS"
bash ensemble_experiments/decorrelated_models/run_stochastic_full_data.sh "$variant"
