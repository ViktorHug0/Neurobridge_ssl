#!/bin/bash
set -euo pipefail

# Convenience wrapper: run few-shot fine-tuning experiment only.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_FEWSHOT_FINETUNING=true
export RUN_PROGRESSIVE=false
export RUN_SPLIT_TRANSFER=false
export RUN_REPETITION_ABLATION=false
export RUN_FEWSHOT=false
export RUN_TRAINSET_TRANSFER=false
export RUN_SUBSPACE_ROTATION=false
export RUN_INDUCTIVE_ROTATION=false

exec bash "${SCRIPT_DIR}/run_rebuttal_suite.sh" "$@"
