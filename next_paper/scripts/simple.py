"""Simplest D-agnostic upgrades of GALA (R3 train distractors, ATM). Each is a one- or two-line change of the loop.
  gala        published loop
  F0.2        reference: start 0.8 SAW + 0.2 moment matching, every step shrunk to identity with lambda 0.2
  shr0.1      reference: SAW start, every step shrunk with lambda 0.1 (SCORE's value)
  avg         iterate averaging: mean over the loop's steps of the z-scored scores (snapshot ensemble)
  w2          gallery whitened with its own shrinkage whitening (SAW params) too; GALA in that space
  anc{l}      anchored Procrustes: every step re-fits the TOTAL rotation from the start,
              R = polar(X0^T P_t G + l ||.||_2 I), t = X0 R  (a prior towards the start, not a step size)
  ancabs{m}   same with an absolute prior mass m * N (shrinks more when the plan is diffuse)
usage: simple.py --seed 3300
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm, procrustes  # noqa: E402
from agnostic import zrow  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def sc(t, gal, p):
    return score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)


def avg_loop(t, gal, p):
    s = sc(t, gal, p)
    acc = zrow(s)
    for _ in range(int(p.soft_procrustes_steps)):
        t = apply_orthogonal_map(t, procrustes(t, gal, damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), p, 0.0))
        s = sc(t, gal, p)
        acc = acc + zrow(s)
    return acc


def anchored(x0, gal, p, lam=0.0, mass=None):
    gn, t = nrm(gal), x0
    s = sc(t, gal, p)
    for _ in range(int(p.soft_procrustes_steps)):
        c = x0.T @ np.power(np.clip(damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), 0, None), p.soft_procrustes_power) @ gn
        prior = mass * len(x0) if mass is not None else lam * np.linalg.norm(c, 2)
        u, _, vt = np.linalg.svd(c + prior * np.eye(len(c)))
        t = nrm(x0 @ (u @ vt))
        s = sc(t, gal, p)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/simple_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            res = {'gala': loop(x, gal, p)[0],
                   'F0.2': loop(nrm(0.8 * x + 0.2 * nrm(mm(q, gal))), gal, p, lam=0.2)[0],
                   'shr0.1': loop(x, gal, p, lam=0.1)[0],
                   'avg': avg_loop(x, gal, p),
                   'w2': loop(x, nrm(fit_saw_transform(gal, p)[0]), p)[0]}
            for lam in (0.0, 0.1, 0.3, 1.0):
                res[f'anc{lam}'] = anchored(x, gal, p, lam=lam)
            for m in (0.01, 0.02, 0.04):
                res[f'ancabs{m}'] = anchored(x, gal, p, mass=m)
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
