"""GALA = one large coarse rotation (step 1, |M - I| ~ 0.85, helps at every D) + small refinements (help at D = 0, hurt
with distractors). Keep the coarse step, make only the refinement robust. All D-agnostic.
  gala         published loop
  F0.2         reference (blend start 0.2 + step shrinkage 0.2)
  jc{l}        jump-then-crawl: step 1 unshrunk, steps 2..16 shrunk with lambda l (SAW start)
  jcF{l}       same from the F0.2 start (0.8 SAW + 0.2 moment matching)
  mnn{k}       step 1 as GALA, then k Procrustes refinements on CSLS(p.csls_k) mutual nearest neighbours only
  mnnw{k}      same, pairs weighted by their top-1 / top-2 CSLS margin (SCORE / Conneau-style)
usage: coarse_fine.py --seed 3300
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm, procrustes  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def sc(t, gal, p):
    return score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)


def step(t, gal, s, p, lam):
    return apply_orthogonal_map(t, procrustes(t, gal, damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), p, lam))


def jump_crawl(t, gal, p, lam):
    s = sc(t, gal, p)
    for i in range(int(p.soft_procrustes_steps)):
        t = step(t, gal, s, p, 0.0 if i == 0 else lam)
        s = sc(t, gal, p)
    return s


def mnn_refine(t, gal, p, k, weighted):
    gn = nrm(gal)
    s = sc(t, gal, p)
    t = step(t, gal, s, p, 0.0)  # GALA's coarse step
    s = sc(t, gal, p)
    for _ in range(k):
        fwd, bwd = s.argmax(1), s.argmax(0)
        mut = np.flatnonzero(bwd[fwd] == np.arange(len(t)))
        if len(mut) < 2:
            break
        top2 = np.sort(s[mut], 1)[:, -2:]
        wt = top2[:, 1] - top2[:, 0] if weighted else np.ones(len(mut))
        u, _, vt = np.linalg.svd((t[mut] * wt[:, None]).T @ gn[fwd[mut]])
        t = nrm(t @ (u @ vt))
        s = sc(t, gal, p)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/coarsefine_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            xf = nrm(0.8 * x + 0.2 * nrm(mm(q, gal)))
            res = {'gala': loop(x, gal, p)[0], 'F0.2': loop(xf, gal, p, lam=0.2)[0]}
            for lam in (0.3, 1.0, 3.0):
                res[f'jc{lam}'] = jump_crawl(x, gal, p, lam)
                res[f'jcF{lam}'] = jump_crawl(xf, gal, p, lam)
            for k in (1, 3, 10):
                res[f'mnn{k}'] = mnn_refine(x, gal, p, k, False)
                res[f'mnnw{k}'] = mnn_refine(x, gal, p, k, True)
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
