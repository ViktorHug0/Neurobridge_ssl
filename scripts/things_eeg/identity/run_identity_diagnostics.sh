#!/usr/bin/env bash
# Identity-Trap diagnostics (variance decomposition + WSCI) on ONE LOSO checkpoint.
# Embeds all 10 subjects' 200-way test trials into that checkpoint's alignment
# space; the held-out subject (from its train_config) is the unseen outlier.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# current-best base model (param_k30, featdim512, seed3300), held-out sub-01.
# Override: CKPT=... DEVICE=... bash scripts/things_eeg/identity/run_identity_diagnostics.sh
CKPT="${CKPT:-results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-170207/param_k30_pool51_do050_featdim512_seed3300/20260429-170229-sub-01}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-results/things_eeg/identity_diag}"
OUT_NAME="${OUT_NAME:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"

python scripts/things_eeg/identity/identity_diagnostics.py \
  --checkpoint_dir "$CKPT" \
  --device "$DEVICE" \
  --output_dir "$OUT_DIR" \
  --output_name "$OUT_NAME"

echo "-> $OUT_DIR/$OUT_NAME.json (+ .png)"
