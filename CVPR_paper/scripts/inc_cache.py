"""Cache the query pools of the incremental protocol: for each subject and each q, the 80/q disjoint q-repetition
averages of every test image, encoded by the frozen LOSO encoder (same blocks as stream_grid.py: seed 1000 + sid).
Writes <cache_dir>/sub-XX.npz with g (200, d) and q5/q10/q20/q40/q80 (n_blocks, 200, d); with the 12 test repetitions
of THINGS-MEG, q1/q2/q3/q4/q6/q12. INC_EEG_DIR / INC_IMG_DIR replace the data paths stored with the checkpoint (runs
trained on another machine).
usage: SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth inc_cache.py <loso run dir> <cache_dir> [subjects...]
"""
import os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
import shared  # noqa: E402
from shared import encode_repetition_blocks, load_subject_context  # noqa: E402

REPS = {80: (5, 10, 20, 40, 80), 12: (1, 2, 3, 4, 6, 12)}  # by number of test repetitions
_args = shared.build_eval_args


def _moved(*a):
    e = _args(*a)
    e.eeg_data_dir, e.image_feature_dir = os.environ.get('INC_EEG_DIR', e.eeg_data_dir), os.environ.get('INC_IMG_DIR', e.image_feature_dir)
    return e


shared.build_eval_args = _moved

if __name__ == '__main__':
    src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for sid in [int(s) for s in sys.argv[3:]] or range(1, 11):
        path = f'{out}/sub-{sid:02d}.npz'
        if os.path.exists(path):
            continue
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(src, rt, sid, average=False)
        z = {}
        n = np.asarray(ds.eeg_data_list[0]).shape[2]
        for q in REPS[n]:
            blocks, g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[q] * (n // q), seed=1000 + sid)
            z[f'q{q}'], z['g'] = np.stack(blocks), g
        np.savez(path, **z)
        print(sid, {k: v.shape for k, v in z.items()}, flush=True)
