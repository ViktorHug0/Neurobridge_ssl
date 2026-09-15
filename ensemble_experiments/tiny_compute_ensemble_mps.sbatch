#!/bin/bash
# Resume Tiny-10 from logical task 24, colocating three independent folds on
# each RTX 3080 under NVIDIA MPS. Read-only EEG caches are memory-mapped, so the
# processes share host pages while retaining separate models/optimizers/seeds.
#SBATCH --job-name=tiny10_mps3
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-25%3
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-mps3-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-mps3-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly FIRST_LOGICAL_TASK=${FIRST_LOGICAL_TASK:-24}
readonly LAST_LOGICAL_TASK=99
readonly PACK_SIZE=3
readonly FIRST_TASK=$((FIRST_LOGICAL_TASK + PACK_SIZE * SLURM_ARRAY_TASK_ID))
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_compute_ensemble/mps3_logs/$SLURM_ARRAY_JOB_ID"
readonly MEMBER_ORDER=(0 5 1 6 2 7 3 8 4 9)

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
  local position=$((logical_task % 10))
  local member=${MEMBER_ORDER[$position]}
  local target=$((logical_task / 10 + 1))
  local task_log="$LOG_ROOT/task-$(printf '%02d' "$logical_task").log"

  echo "[mps3] logical_task=$logical_task member=$member target=$target log=$task_log"
  (
    export MEMBER="$member"
    export TARGET="$target"
    export DEVICE=cuda:0
    export NW=2
    bash ensemble_experiments/run_tiny_compute_ensemble_fold.sh
  ) >"$task_log" 2>&1
}

pids=()
for offset in $(seq 0 $((PACK_SIZE - 1))); do
  logical_task=$((FIRST_TASK + offset))
  if [ "$logical_task" -le "$LAST_LOGICAL_TASK" ]; then
    run_logical_task "$logical_task" &
    pids+=("$!")
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

exit "$status"
