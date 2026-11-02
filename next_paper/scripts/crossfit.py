"""Can held-out (cross-fitted) predictive fit choose how much rotation to trust, without D or N/M?
Family F(r), r in [0, 1]: start = (1-r) SAW + r * moment matching to the gallery, every Procrustes step shrunk to the
identity with lambda = r (r = 0 is GALA, r = 1 the conservative expert; robust_NM = F(1 - N/M)).
For each r: true top-1 of F(r) fitted on all queries, and label-free criteria of the rotation fitted on half the
queries (A), evaluated on the other half (B), 2 folds x `--splits` random splits (start transforms use all queries:
they are correspondence-free). In-sample versions of the same criteria for contrast.
  maxcos   mean_i max_j cos(T x_i, g_j)
  lse{k}   mean_i log sum_j exp(k cos(T x_i, g_j))        (vMF mixture log-likelihood up to a constant)
  hung     mean cos of the Hungarian matching of B to the gallery
  zmax     mean_i per-query z-score of the max cosine (scale-free: comparable across starts)
  zmarg    mean_i per-query z-scored top-1 minus top-2 margin
  cslsmax  mean_i max_j CSLS(10) (gallery hub term over the evaluated queries)
usage: crossfit.py --seed 3300 [--arm atm128_vc]
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from stability import recover  # noqa: E402
from shared import fit_saw_transform  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from label_free import csls  # noqa: E402

RS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
KS = (10, 30, 100)


def crit(t, gn):
    c = t @ gn.T
    out = {'maxcos': c.max(1).mean(), 'hung': c[np.arange(len(c)), linear_sum_assignment(-c)[1]].mean()}
    for k in KS:
        out[f'lse{k}'] = logsumexp(k * c, 1).mean()
    zc = (c - c.mean(1, keepdims=True)) / c.std(1, keepdims=True)
    top2 = np.sort(zc, 1)[:, -2:]
    out['zmax'], out['zmarg'] = top2[:, 1].mean(), (top2[:, 1] - top2[:, 0]).mean()
    out['cslsmax'] = csls(t, gn, 10).max(1).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--splits', type=int, default=3)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/crossfit_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'r', 'metric', 'value'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            gn = nrm(gal)
            for r in RS:
                s0 = nrm((1 - r) * x + r * nrm(mm(q, gal)))
                s, t, _ = loop(s0, gal, p, lam=r)
                w.writerow([sid, D, r, 'top1', topk_acc(s, y)[1]])
                for k, v in crit(t, gn).items():
                    w.writerow([sid, D, r, f'in_{k}', v])
                acc = {}
                for sp in range(a.splits):
                    perm = np.random.default_rng(1000 * sid + sp).permutation(200)
                    for A, B in ((perm[:100], perm[100:]), (perm[100:], perm[:100])):
                        tA = loop(s0[A], gal, p, lam=r)[1]
                        m = recover(s0[A], tA)
                        for k, v in crit(nrm(s0[B] @ m), gn).items():
                            acc.setdefault(k, []).append(v)
                for k, v in acc.items():
                    w.writerow([sid, D, r, f'cv_{k}', float(np.mean(v))])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
