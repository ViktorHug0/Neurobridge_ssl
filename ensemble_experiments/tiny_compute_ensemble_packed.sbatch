#!/bin/bash
# Resume Tiny-10 from logical task 18, packing two independent folds per GPU.
# The paired tasks alternate TinyTSConv/TinyATM, so their kernels and memory
# footprints are complementary. The training recipe itself is unchanged.
#SBATCH --job-name=tiny10_pack
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-40%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-pack-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny10-pack-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly FIRST_LOGICAL_TASK=${FIRST_LOGICAL_TASK:-18}
readonly LAST_LOGICAL_TASK=99
readonly TASK_A=$((FIRST_LOGICAL_TASK + 2 * SLURM_ARRAY_TASK_ID))
readonly TASK_B=$((TASK_A + 1))
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_compute_ensemble/packed_logs/$SLURM_ARRAY_JOB_ID"
readonly MEMBER_ORDER=(0 5 1 6 2 7 3 8 4 9)

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

mps_enabled=0
if [ "${ENABLE_MPS:-0}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/neurobridge-mps-$SLURM_JOB_ID"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/neurobridge-mps-log-$SLURM_JOB_ID"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if nvidia-cuda-mps-control -d; then
    mps_enabled=1
    echo "[packed] NVIDIA MPS enabled: $CUDA_MPS_PIPE_DIRECTORY"
  else
    echo "[packed] NVIDIA MPS unavailable; continuing with ordinary CUDA scheduling" >&2
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  fi
fi

cleanup_mps() {
  if [ "$mps_enabled" -eq 1 ]; then
    echo quit | nvidia-cuda-mps-control || true
  fi
}
trap cleanup_mps EXIT

run_logical_task() {
  local logical_task=$1
  local position=$((logical_task % 10))
  local member=${MEMBER_ORDER[$position]}
  local target=$((logical_task / 10 + 1))
  local task_log="$LOG_ROOT/task-$(printf '%02d' "$logical_task").log"

  echo "[packed] logical_task=$logical_task member=$member target=$target log=$task_log"
  (
    export MEMBER="$member"
    export TARGET="$target"
    export DEVICE=cuda:0
    # Two loaders share the node's eight CPUs. Three workers per fold left the
    # calibration GPU-bound while avoiding CPU oversubscription.
    export NW=3
    bash ensemble_experiments/run_tiny_compute_ensemble_fold.sh
  ) >"$task_log" 2>&1
}

run_logical_task "$TASK_A" &
pid_a=$!

pid_b=
if [ "$TASK_B" -le "$LAST_LOGICAL_TASK" ]; then
  run_logical_task "$TASK_B" &
  pid_b=$!
fi

status=0
wait "$pid_a" || status=$?
if [ -n "$pid_b" ]; then
  wait "$pid_b" || status=$?
fi

exit "$status"
