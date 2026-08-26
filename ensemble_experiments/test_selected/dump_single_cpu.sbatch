#!/bin/bash
# Export one or more completed folds without consuming a training GPU.
# Submit with, for example:
#   sbatch --export=ALL,ARM=atm_iv_group_e100,SUBJECTS=1 dump_single_cpu.sbatch
#
# ARM and SUBJECTS are intentionally required so this helper cannot silently
# export the wrong experiment when it is reused.
#SBATCH --job-name=ens50_cpu_dump
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --nodelist=sl-mee-br-502
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-cpu-dump-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-cpu-dump-%j.err
set -euo pipefail

: "${ARM:?submit with --export=ALL,ARM=<arm>,SUBJECTS=<space-separated folds>}"
: "${SUBJECTS:?submit with --export=ALL,ARM=<arm>,SUBJECTS=<space-separated folds>}"

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""

extra_args=()
if [[ -n ${DUMP_SUFFIX:-} ]]; then
  extra_args+=(--dump-suffix "$DUMP_SUFFIX")
fi
if [[ -n ${EEG_TTA_SHIFTS:-} ]]; then
  extra_args+=(--eeg-tta-shifts $EEG_TTA_SHIFTS)
fi

python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" \
  --run-root "${RUN_ROOT:-results/things_eeg/ensemble50_testselected/$ARM/seed3300}" \
  --subjects $SUBJECTS --device cpu "${extra_args[@]}"
