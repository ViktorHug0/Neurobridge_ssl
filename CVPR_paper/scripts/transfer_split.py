"""Does SAGE's correction transfer to images it was not fitted on, at the same SNR? (v2 paper, user 2026-10-24)
The 200 test images (80-rep queries) are split in two halves A / B (5 random splits x 2 directions). On B (100-way):
  zero_csls      CSLS on the raw queries                 refit          SAGE fitted on B itself (transductive)
  white_frozen   whitening estimated on A, applied to B  sage_frozen    SAGE (whitening + rotation) fitted on A, frozen
usage: transfer_split.py <params_csv> <cache_dir> <out_csv>
"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, fit_tta_calibration, score_features  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def main():
    params_csv, cache, out = sys.argv[1:4]
    rows = []
    for sid in range(1, 11):
        z = np.load(f'{cache}/sub-{sid:02d}.npz')
        q, g = z['q'], z['g']
        p = subject_params(params_csv, sid)
        acc = lambda x, gg: 100 * (score_features(x, gg, use_csls=True, csls_k=p.csls_k).argmax(1) == np.arange(len(x))).mean()  # noqa: E731
        for split in range(5):
            perm = np.random.default_rng(1000 * sid + split).permutation(200)
            for a, b in ((perm[:100], perm[100:]), (perm[100:], perm[:100])):
                ca = fit_tta_calibration(q[a], g[a], p)
                white = dict(ca, orthogonal_map=np.eye(len(ca['orthogonal_map']), dtype=np.float32))
                rows.append({'subject': sid, 'split': split, 'zero_csls': acc(q[b], g[b]),
                             'white_frozen': acc(apply_tta_calibration(q[b], white), g[b]),
                             'sage_frozen': acc(apply_tta_calibration(q[b], ca), g[b]),
                             'refit': acc(apply_tta_calibration(q[b], fit_tta_calibration(q[b], g[b], p)), g[b])})
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(d.groupby('subject').mean().drop(columns='split').mean().round(1))


if __name__ == '__main__':
    main()
