#!/bin/bash
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
if [ "$(hostname)" != sl-tp-br-513 ]; then
    echo 'Only sl-tp-br-513 is authorized.' >&2
    exit 1
fi
if tmux has-session -t compact-followup 2>/dev/null; then
    echo 'Inspect existing compact-followup tmux session before relaunching.' >&2
    exit 1
fi
tmux new-session -d -s compact-followup -c /nasbrain/p20fores/Neurobridge_SSL \
    "bash -c 'source .venv/bin/activate; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; python -u -m ensemble_experiments.compact_followup.run_queue --arm electrode >> results/things_eeg/compact_followup_20260908/logs/queue_electrode.log 2>&1'"
tmux set-option -t compact-followup remain-on-exit on
echo 'Started electrode-only queue in tmux: compact-followup'
