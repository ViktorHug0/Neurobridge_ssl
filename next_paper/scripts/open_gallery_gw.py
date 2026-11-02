"""Relational support selection for R3: can the second-order structure find the 200 present images among 200 + D?
Partial entropic Gromov-Wasserstein (inner-product form), our own solver: T (N x M) with rows = 1 and columns <= 1
(injective), maximising alpha * <S, T> + (1 - alpha) * <Cq T Cg, T> by mirror ascent: T <- capped-Sinkhorn(G / tau),
G = alpha * S + (1 - alpha) * Cq T Cg, each term standardised. Cq, Cg = cosine similarities of the centred clouds
(the matrices behind RSA; rotation-invariant). S = CSLS scores from the robust start (blend of SAW and moment matching).
Support = the N columns with the most mass. Reported: support precision (fraction of selected columns that are true
test images) and top-1 of robust GALA fitted on the support and scored on the full gallery.
usage: open_gallery_gw.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import capped, loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def partial_gw(s, cq, cg, alpha, tau, iters=30):
    n, m = s.shape
    caps = np.ones((1, m))
    t = capped(zs(s), tau, 12, caps)
    for _ in range(iters):
        g = alpha * zs(s) + (1 - alpha) * zs(cq @ (t / n) @ cg)
        t = capped(g, tau, 12, caps)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/gw_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'support_precision', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = z['q'].astype(np.float32), z['g'].astype(np.float32), z['d'].astype(np.float32)
        cq = nrm(q - q.mean(0)) @ nrm(q - q.mean(0)).T
        for D in [d for d in D_LIST if d > 0]:
            gal = np.concatenate([g, dis[:D]], 0)
            n, m = len(q), len(gal)
            r = 1 - n / m
            start = nrm((1 - r) * nrm(fit_saw_transform(q, p)[0]) + r * nrm(mm(q, gal)))
            s = score_features(start, gal, use_csls=p.use_csls, csls_k=p.csls_k)
            gc = nrm(gal - gal.mean(0))
            cg = gc @ gc.T
            sel = {'hung': np.sort(linear_sum_assignment(-s)[1])}
            for alpha in (0.0, 0.3, 0.7):
                for tau in (0.1, 0.3):
                    t = partial_gw(s, cq, cg, alpha, tau)
                    sel[f'gw_a{alpha}_t{tau}'] = np.sort(np.argsort(-t.sum(0))[:n])
            for name, cols in sel.items():
                _, tt, _ = loop(start, gal[cols], p)
                acc = topk_acc(score_features(tt, gal, use_csls=p.use_csls, csls_k=p.csls_k), y)[1]
                w.writerow([sid, D, name, 100 * (cols < 200).mean(), acc])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
