"""Can robust estimation recover the rotation from contaminated correspondences (R3)? D-agnostic except the oracle.
Correspondences = rectangular Hungarian on a start score matrix (fusion G + 3 C1). Rotation fitted (in SAW space,
shrinkage-free) on these pairs by:
  ls        least squares on all pairs (plain Procrustes)
  oracle    least squares on the TRUE pairs among them only (ideal robust estimator; uses labels)
  cauchy    IRLS with Cauchy weights w = 1 / (1 + (e / c)^2), e = pair residual, c = 1.4826 * MAD(e); 10 rounds
  gm        IRLS with Geman-McClure weights w = 1 / (1 + (e / c)^2)^2
then each rotated query set is scored with CSLS on the full gallery; and 'iter_*' = 3 outer rounds re-matching with
Hungarian on the new scores. Reports pair precision, top-1.
usage: robust_procrustes.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def wproc(x, y, w):
    u, _, vt = np.linalg.svd((x * w[:, None]).T @ y)
    return u @ vt


def irls(x, y, kind, rounds=10):
    w = np.ones(len(x))
    for _ in range(rounds):
        r = wproc(x, y, w)
        e = np.linalg.norm(x @ r - y, axis=1)
        c = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9
        w = 1 / (1 + (e / c) ** 2)
        if kind == 'gm':
            w = w ** 2
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/robust_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'metric', 'value'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            gn = nrm(gal)
            s0 = zrow(loop(x, gal, p)[0]) + 3 * zrow(loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0])
            cols = linear_sum_assignment(-s0)[1]
            w.writerow([sid, D, 'pairs', 'precision', 100 * (cols == y).mean()])
            fits = {'ls': wproc(x, gn[cols], np.ones(200)), 'oracle': wproc(x[cols == y], gn[cols][cols == y], np.ones((cols == y).sum())),
                    'cauchy': irls(x, gn[cols], 'cauchy'), 'gm': irls(x, gn[cols], 'gm')}
            for name, r in fits.items():
                w.writerow([sid, D, name, 'top1', topk_acc(score_features(nrm(x @ r), gal, use_csls=p.use_csls, csls_k=p.csls_k), y)[1]])
            for kind in ('cauchy', 'gm'):  # outer re-matching
                r = fits[kind]
                for _ in range(3):
                    s = score_features(nrm(x @ r), gal, use_csls=p.use_csls, csls_k=p.csls_k)
                    c2 = linear_sum_assignment(-s)[1]
                    r = irls(x, gn[c2], kind)
                w.writerow([sid, D, f'iter_{kind}', 'top1', topk_acc(score_features(nrm(x @ r), gal, use_csls=p.use_csls, csls_k=p.csls_k), y)[1]])
                w.writerow([sid, D, f'iter_{kind}', 'precision', 100 * (c2 == y).mean()])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
