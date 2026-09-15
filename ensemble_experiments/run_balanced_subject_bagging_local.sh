#!/bin/bash
# Persistent, resumable serial runner for one allocated GPU.
set -uo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/subject_cohort_bagging/testselected_balanced7x8}
EXPECTED_HOST=${EXPECTED_HOST:-sl-tp-br-513}
MAX_PASSES=${MAX_PASSES:-3}
TASK_START=${TASK_START:-0}
TASK_STRIDE=${TASK_STRIDE:-1}
PHYSICAL_CUDA_DEVICE=${PHYSICAL_CUDA_DEVICE:-0}
RUNNER_NAME=${RUNNER_NAME:-runner}

cd "$REPO_ROOT"
source .venv/bin/activate
mkdir -p "$RESULT_ROOT/local_runner"
master_log="$RESULT_ROOT/local_runner/$RUNNER_NAME.log"
exec >> "$master_log" 2>&1

echo "[$(date --iso-8601=seconds)] starting host=$(hostname) pid=$$ result_root=$RESULT_ROOT task_start=$TASK_START task_stride=$TASK_STRIDE physical_cuda=$PHYSICAL_CUDA_DEVICE"
if [ "$(hostname)" != "$EXPECTED_HOST" ]; then
  echo "refusing to run: expected $EXPECTED_HOST, got $(hostname)" >&2
  exit 2
fi

if [ "$TASK_START" -lt 0 ] || [ "$TASK_START" -gt 79 ] || [ "$TASK_STRIDE" -le 0 ]; then
  echo "invalid task shard: TASK_START=$TASK_START TASK_STRIDE=$TASK_STRIDE" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_CUDA_DEVICE"
export DEVICE=cuda:0
export NW=${NW:-6}
export RESULT_ROOT

remaining_count() {
  local remaining=0 target member dump
  for task in $(seq "$TASK_START" "$TASK_STRIDE" 79); do
    target=$((task / 8 + 1))
    member=$((task % 8))
    dump=$(printf '%s/target-%02d/member-%d/embeddings.npz' \
      "$RESULT_ROOT" "$target" "$member")
    if [ ! -f "$dump" ]; then
      remaining=$((remaining + 1))
    fi
  done
  echo "$remaining"
}

for pass in $(seq 1 "$MAX_PASSES"); do
  echo "[$(date --iso-8601=seconds)] pass=$pass remaining=$(remaining_count)"
  for task in $(seq "$TASK_START" "$TASK_STRIDE" 79); do
    export TARGET=$((task / 8 + 1))
    export MEMBER=$((task % 8))
    dump=$(printf '%s/target-%02d/member-%d/embeddings.npz' \
      "$RESULT_ROOT" "$TARGET" "$MEMBER")
    if [ -f "$dump" ]; then
      continue
    fi

    echo "[$(date --iso-8601=seconds)] launch task=$task target=$TARGET member=$MEMBER"
    if bash ensemble_experiments/run_balanced_subject_bag_fold.sh; then
      echo "[$(date --iso-8601=seconds)] complete task=$task target=$TARGET member=$MEMBER"
    else
      status=$?
      echo "[$(date --iso-8601=seconds)] failed status=$status task=$task target=$TARGET member=$MEMBER" >&2
    fi
  done

  remaining=$(remaining_count)
  if [ "$remaining" -eq 0 ]; then
    python -m ensemble_experiments.analyze_balanced_subject_bagging \
      --result-root "$RESULT_ROOT"
    echo "[$(date --iso-8601=seconds)] assigned shard complete"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] pass=$pass ended remaining=$remaining"
done

python -m ensemble_experiments.analyze_balanced_subject_bagging \
  --result-root "$RESULT_ROOT"
echo "[$(date --iso-8601=seconds)] stopped with $(remaining_count) members incomplete" >&2
exit 1
