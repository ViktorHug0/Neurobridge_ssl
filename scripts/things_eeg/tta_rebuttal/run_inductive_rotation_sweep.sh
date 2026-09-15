#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export RUN_PROGRESSIVE=false
export RUN_SPLIT_TRANSFER=false
export RUN_REPETITION_ABLATION=false
export RUN_FEWSHOT=false
export RUN_TRAINSET_TRANSFER=false
export RUN_SUBSPACE_ROTATION=false
export RUN_INDUCTIVE_ROTATION=true
export RUN_BOTH_MODELS=false
export MODEL_TAG=featdim512
export INDUCTIVE_SEEDS="${INDUCTIVE_SEEDS:-3300}"
export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/things_eeg/tta_rebuttal/inductive_rotation_20260603-115608}"

exec bash "${SCRIPT_DIR}/run_rebuttal_suite.sh"
