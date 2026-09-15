#!/bin/bash
# Ten targets x three full-source, balanced soft-weight TSConvs.
# Each arm sees all nine sources; group weights rotate through 1.3/1.0/0.7.
#SBATCH --job-name=softbag_ts
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-29%3
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-softbag-ts-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-softbag-ts-%A_%a.err
set -euo pipefail

export TARGET=$((SLURM_ARRAY_TASK_ID / 3 + 1))
export COHORT=$((SLURM_ARRAY_TASK_ID % 3))
export GROUP_MODE=soft_weighted
export MODEL_SEED=$((3300 + COHORT))
export WEIGHT_DELTA=0.3
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_cohort_bagging/testselected_softweights_d030
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}

cd /nasbrain/p20fores/Neurobridge_SSL
echo "[softbag-array] task=$SLURM_ARRAY_TASK_ID target=$TARGET arm=$COHORT seed=$MODEL_SEED"
bash ensemble_experiments/run_subject_cohort_fold.sh
