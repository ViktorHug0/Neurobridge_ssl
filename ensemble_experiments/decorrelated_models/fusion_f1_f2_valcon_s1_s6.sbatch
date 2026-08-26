#!/bin/bash
# F1/F2 fusion-aware ATM+TSConv pilot on subjects 1--6.
# Four lanes: each beta setting is split into subjects 1--3 and 4--6.
#SBATCH --job-name=fusion_f12
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-fusion-f12-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-fusion-f12-%A_%a.err
set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:?array task id is required}" in
  0)
    export ARM=fusion_valcon_atm_tsconv_b010_g000_bs512
    export BETA_ENSEMBLE=0.10
    export SUBJECTS="1 2 3"
    ;;
  1)
    export ARM=fusion_valcon_atm_tsconv_b010_g000_bs512
    export BETA_ENSEMBLE=0.10
    export SUBJECTS="4 5 6"
    ;;
  2)
    export ARM=fusion_valcon_atm_tsconv_b030_g000_bs512
    export BETA_ENSEMBLE=0.30
    export SUBJECTS="1 2 3"
    ;;
  3)
    export ARM=fusion_valcon_atm_tsconv_b030_g000_bs512
    export BETA_ENSEMBLE=0.30
    export SUBJECTS="4 5 6"
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

cd /nasbrain/p20fores/Neurobridge_SSL
export RESULT_ROOT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/decorrelated_models/fusion_wave
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}
export BATCH_SIZE=512
export NUM_EPOCHS=50
export LAMBDA_DIV=0
export GAMMA_RESCUE=0
export RESCUE_TEMPERATURE=0.5
export RESCUE_START_EPOCH=1
export SELECTION_PROTOCOL=valcon
export VAL_CONCEPT_RATIO=0.10
export VAL_CONCEPT_SEED=20260822

echo "[fusion-pilot] task=${SLURM_ARRAY_TASK_ID} arm=${ARM} beta=${BETA_ENSEMBLE} subjects=${SUBJECTS}"
bash ensemble_experiments/decorrelated_models/run_atm_tsconv_l010.sh
