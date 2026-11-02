#!/bin/bash
# Sequentially read the fp16 training/test memmaps of a cv4 *_e1_things run into this node's page cache
# (stage_eeg_cache skips warming above half the memory limit; 18 GiB fits a 31 GiB node next to one trainer).
cd /nasbrain/p20fores/Neurobridge_SSL
.venv/bin/python - "$@" <<'PY'
import json, os, sys
sys.path.insert(0, '.')
from module.dataset import _eeg_cache_key, _eeg_cache_path
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
ch = json.load(open(f'{D}/info.json'))['ch_names']
test = [int(s) for s in sys.argv[1:]]
paths = []
for s in range(1, 43):
    for train in ((True,) if s not in test else (False,)):
        p = _eeg_cache_path(D, _eeg_cache_key(subject_id=s, train=train, average=False, selected_channels=ch, time_window=[0, 120]))
        p16 = p[:-4] + '_fp16.npy'
        paths.append(p16 if os.path.exists(p16) else p)
n = 0
for p in paths:
    with open(p, 'rb') as f:
        while (b := f.read(64 << 20)):
            n += len(b)
print(f'warmed {len(paths)} files, {n / 2**30:.1f} GiB', flush=True)
PY
