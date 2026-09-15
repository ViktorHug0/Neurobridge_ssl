#!/bin/bash
# BF16 Tiny reference pair on subjects 1--3. Subject-1 TSConv may already
# exist from the equivalence gate; the runner skips completed exports.
#SBATCH --job-name=tiny40_bf16_s123
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=00:40:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-bf16-s123-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-bf16-s123-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_reference_pair_40e_bf16/logs/$SLURM_ARRAY_JOB_ID"
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

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  logical_tasks=(0 1 2 3)
else
  logical_tasks=(4 5)
fi

run_logical_task() {
  local logical_task=$1
  local target=$((logical_task / 2 + 1))
  local member=$((logical_task % 2))
  (
    export TARGET="$target" MEMBER="$member" DEVICE=cuda:0 NW=1
    export AMP_DTYPE=bfloat16 RUN_TAG=tiny40_bf16
    export RESULT_ROOT=$REPO_ROOT/results/things_eeg/tiny_reference_pair_40e_bf16/testselected_internvit28
    bash ensemble_experiments/run_tiny40_reference_fold.sh
  ) >"$LOG_ROOT/task-${logical_task}.log" 2>&1
}

pids=()
for logical_task in "${logical_tasks[@]}"; do
  run_logical_task "$logical_task" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
