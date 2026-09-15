#!/bin/bash
# Two MPS packs: six independent 40-epoch reference folds on subjects 1--3.
#SBATCH --job-name=tiny40_s123
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=00:45:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-s123-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-s123-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly FIRST_LOGICAL_TASK=$((3 * SLURM_ARRAY_TASK_ID))
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_reference_pair_40e/logs/$SLURM_ARRAY_JOB_ID"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/neurobridge-mps-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/neurobridge-mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
cleanup_mps() {
  echo quit | nvidia-cuda-mps-control || true
}
trap cleanup_mps EXIT

run_logical_task() {
  local logical_task=$1
  local target=$((logical_task / 2 + 1))
  local member=$((logical_task % 2))
  local family=tiny_tsconv
  if [ "$member" -eq 1 ]; then family=tiny_atm; fi
  (
    export TARGET="$target" MEMBER="$member" DEVICE=cuda:0 NW=2
    bash ensemble_experiments/run_tiny40_reference_fold.sh
  ) >"$LOG_ROOT/task-${logical_task}-sub-${target}-${family}.log" 2>&1
}

pids=()
for offset in 0 1 2; do
  run_logical_task "$((FIRST_LOGICAL_TASK + offset))" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
