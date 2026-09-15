#!/bin/bash
# LEACE erasure across all 10 LOSO folds of one config -> the inductive
# subject-axis-erasure retrieval delta as a LOSO average.
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"

# a config dir holding 10 per-subject run dirs (each with checkpoint_test_best.pth)
CONFIG_DIR="${CONFIG_DIR:-results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-170207/param_k30_pool51_do050_featdim512_seed3300}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-results/things_eeg/identity_diag/leace_$(basename "$CONFIG_DIR")}"
mkdir -p "$OUT_DIR"

for ck in $(find "$CONFIG_DIR" -name checkpoint_test_best.pth | sort); do
    RUN_DIR="$(dirname "$ck")"
    NAME="$(basename "$RUN_DIR")"
    echo "### LEACE on $NAME"
    python scripts/things_eeg/identity/leace_erasure.py \
        --checkpoint_dir "$RUN_DIR" --device "$DEVICE" \
        --output_dir "$OUT_DIR" --output_name "$NAME"
done

python3 -c "
import glob, json, os, numpy as np, sys
js = sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))
rows = [json.load(open(f)) for f in js]
p = np.array([r['top1_plain'] for r in rows]); e = np.array([r['top1_erased'] for r in rows])
print(f'\n== LEACE LOSO ({len(rows)} folds) ==')
for r in rows:
    print(f\"  sub-{r['unseen_subject']:>2}: plain {r['top1_plain']*100:5.2f}%  erased {r['top1_erased']*100:5.2f}%  \"
          f\"delta {100*(r['top1_erased']-r['top1_plain']):+5.2f}pp  (subj-probe {r['train_subject_probe_ba_pre']:.2f}->{r['train_subject_probe_ba_post']:.2f})\")
print(f'  AVG    : plain {p.mean()*100:5.2f}%  erased {e.mean()*100:5.2f}%  delta {100*(e-p).mean():+5.2f}pp')
" "$OUT_DIR"
