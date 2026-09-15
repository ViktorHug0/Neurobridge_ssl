#!/bin/bash
# Persistent/resumable two-arm runner for one subject shard on an allocated GPU.
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/things_eeg/decorrelated_models/stochastic_full_data_wave}
EXPECTED_HOST=${EXPECTED_HOST:-sl-tp-br-513}
PHYSICAL_CUDA_DEVICE=${PHYSICAL_CUDA_DEVICE:-0}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5"}
SUBJECTS=${SUBJECTS//,/ }
RUNNER_NAME=${RUNNER_NAME:-runner}
VARIANTS=${VARIANTS:-"control stochastic"}

cd "$REPO_ROOT"
source .venv/bin/activate
mkdir -p "$RESULT_ROOT/local_runner"
exec >> "$RESULT_ROOT/local_runner/$RUNNER_NAME.log" 2>&1

echo "[$(date --iso-8601=seconds)] starting host=$(hostname) pid=$$ subjects=$SUBJECTS physical_cuda=$PHYSICAL_CUDA_DEVICE"
if [ "$(hostname)" != "$EXPECTED_HOST" ]; then
  echo "refusing to run: expected $EXPECTED_HOST, got $(hostname)" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_CUDA_DEVICE"
export DEVICE=cuda:0
export RESULT_ROOT
export NW=${NW:-6}

for variant in $VARIANTS; do
  echo "[$(date --iso-8601=seconds)] starting variant=$variant subjects=$SUBJECTS"
  SUBJECTS="$SUBJECTS" \
    bash ensemble_experiments/decorrelated_models/run_stochastic_full_data.sh "$variant"
  echo "[$(date --iso-8601=seconds)] completed variant=$variant subjects=$SUBJECTS"
done

python -m ensemble_experiments.decorrelated_models.analyze_stochastic_full_data \
  --result-root "$RESULT_ROOT"
echo "[$(date --iso-8601=seconds)] shard complete"
