#!/bin/bash
set -euo pipefail
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
if [ "$(hostname)" != sl-tp-br-513 ]; then
    echo 'This launcher is for the existing allocation on sl-tp-br-513.' >&2
    exit 1
fi
if tmux has-session -t compact-valcon 2>/dev/null; then
    echo 'tmux session compact-valcon already exists; inspect it before relaunching.' >&2
    exit 1
fi
tmux new-session -d -s compact-valcon -c "$REPO_ROOT" \
    "bash -c 'source .venv/bin/activate; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; python -u -m ensemble_experiments.compact_valcon.run_queue --worker 0 >> results/things_eeg/compact_valcon_20260908/logs/worker0.log 2>&1'"
tmux set-option -t compact-valcon remain-on-exit on
echo 'Started worker 0 in tmux: compact-valcon'
