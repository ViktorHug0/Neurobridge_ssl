"""E8.6 is embedding dimension a hidden TTA knob? Post-hoc, on the cached R1 embeddings (80-rep queries, 200 x 200):
project queries AND gallery on the top-k principal axes of the test gallery (label-free: images only, available at
test time), then plain / Hungarian / SCORE / SAGE (each arm's own cross-fold per-subject params). k = full = no PCA.
usage: dim_sweep_pca.py --arms atm_s3300 rmix_s3300 ... [--ks 8 16 32 64 128 256 512]
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, fit_tta_calibration, score_features  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402

C, S = 'results/things_eeg/cvpr/emb_cache', 'results/things_eeg/cvpr/tta_screen_loso'


def acc(s):
    return 100 * (np.asarray(s).argmax(1) == np.arange(len(s))).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--ks', nargs='+', type=int, default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument('--output_csv', default='results/things_eeg/cvpr/analysis/dim_sweep_pca.csv')
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        params_csv = f'{S}/{arm}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
        for sid in range(1, 11):
            z = np.load(f'{C}/{arm}_b/sub-{sid:02d}.npz')
            q0, g0 = z['q'].astype(np.float32), z['g'].astype(np.float32)
            p = subject_params(params_csv, sid)
            mu = g0.mean(0)
            _, _, vt = np.linalg.svd(g0 - mu, full_matrices=False)
            for k in [k for k in a.ks if k < g0.shape[1]] + [g0.shape[1]]:
                if k < g0.shape[1]:  # ponytail: rank of a 200-image gallery caps k at 199
                    if k >= len(g0):
                        continue
                    B = vt[:k].T
                    q, g = (q0 - mu) @ B, (g0 - mu) @ B
                else:
                    q, g = q0, g0
                s = score_features(q, g)
                r, c = linear_sum_assignment(-s)
                sage = score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True, csls_k=p.csls_k)
                rows.append(dict(arm=arm, subject=sid, k=k, full=k == g0.shape[1], plain=acc(s),
                                 hungarian=100 * (c[np.argsort(r)] == np.arange(len(s))).mean(),
                                 score=acc(score_deploy(q, g)), sage=acc(sage)))
        print(arm, 'done', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.output_csv, index=False)
    print(d.groupby(['arm', 'k'])[['plain', 'hungarian', 'score', 'sage']].mean().round(1).to_string())


if __name__ == '__main__':
    main()
