#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
SESSION_NAME=eeg-ensemble-mechanisms
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Already running: tmux attach -t $SESSION_NAME"
    exit 0
fi
mkdir -p results/things_eeg/ensemble_mechanism_20260907/logs
# UUID avoids the different CUDA and nvidia-smi numeric device order on this host.
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
    "bash -lc 'source .venv/bin/activate; export CUDA_VISIBLE_DEVICES=GPU-bf704308-654f-501f-3a82-1f6b38a4fb18 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; python -u -m ensemble_experiments.mechanism_study.run_queue 2>&1 | tee -a results/things_eeg/ensemble_mechanism_20260907/logs/queue.log; exec bash'"
echo "Started: tmux attach -t $SESSION_NAME"
