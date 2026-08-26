#!/bin/bash
# Five-arm rescue wave, flattened to 50 independent folds for balanced scheduling.
# Submit in four dependency lanes with one active fold per lane.
#SBATCH --job-name=rescue_night
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-rescue-night-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-rescue-night-%A_%a.err
set -euo pipefail

task=${SLURM_ARRAY_TASK_ID:?array task id is required}
arm_index=$((task / 10))
held_subject=$((task % 10 + 1))

case "$arm_index" in
  0)
    export ARM=rescue_atm_tsconv_r3_g050_b0_bs512
    export GAMMA_RESCUE=0.50
    export SELECTION_PROTOCOL=test
    export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/heterogeneous_wave
    ;;
  1)
    export ARM=rescue_valcon_atm_tsconv_g000_b0_bs512
    export GAMMA_RESCUE=0
    export SELECTION_PROTOCOL=valcon
    export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/valcon_wave
    ;;
  2)
    export ARM=rescue_valcon_atm_tsconv_g010_b0_bs512
    export GAMMA_RESCUE=0.10
    export SELECTION_PROTOCOL=valcon
    export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/valcon_wave
    ;;
  3)
    export ARM=rescue_valcon_atm_tsconv_g030_b0_bs512
    export GAMMA_RESCUE=0.30
    export SELECTION_PROTOCOL=valcon
    export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/valcon_wave
    ;;
  4)
    export ARM=rescue_valcon_atm_tsconv_g050_b0_bs512
    export GAMMA_RESCUE=0.50
    export SELECTION_PROTOCOL=valcon
    export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/valcon_wave
    ;;
  *)
    echo "unexpected task $task (arm index $arm_index)" >&2
    exit 2
    ;;
esac

cd /nasbrain/p20fores/Neurobridge_SSL
export NW=${SLURM_CPUS_PER_TASK:-6}
export DEVICE=cuda:0
export BATCH_SIZE=512
export LAMBDA_DIV=0
export BETA_ENSEMBLE=0
export RESCUE_TEMPERATURE=0.5
export RESCUE_START_EPOCH=1
export VAL_CONCEPT_RATIO=0.10
export VAL_CONCEPT_SEED=20260822
export SUBJECTS=$held_subject

echo "[night-wave] task=$task arm_index=$arm_index held=$held_subject"
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
