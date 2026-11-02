"""Contrastive barycentre: GALA's Procrustes target for query i is the plan barycentre b_i = sum_j P_ij g_j; with many
distractors it is mostly a local gallery-density pull (distribution level). Subtract the background barycentre of the
same scores at a c-times higher temperature: target_i = b_i(tau) - b_i(c tau). D-agnostic, one line.
  gala, F0.2           references
  ctr{c}               contrastive target at every step, SAW start
  ctrF{c}              same from the F0.2 start with its step shrinkage (lambda 0.2)
  conf                 GALA with each query's Procrustes row weighted by its plan sharpness 1 / n_eff_i (normalised)
usage: contrastive.py --seed 3300
"""
import argparse, csv, os, sys

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def run(t, gal, p, lam=0.0, c=None, conf=False):
    gn = nrm(gal) if p.soft_procrustes_normalize_inputs else gal  # as GALA's procrustes()
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    for _ in range(int(p.soft_procrustes_steps)):
        a = np.power(np.clip(damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), 0, None), p.soft_procrustes_power)
        b = a @ gn
        if c is not None:
            bg = np.power(np.clip(damped_assign(s, c * p.sinkhorn_tau, p.sinkhorn_iters, 1.0), 0, None), p.soft_procrustes_power)
            b = b - bg @ gn * (a.sum(1, keepdims=True) / np.clip(bg.sum(1, keepdims=True), 1e-30, None))  # same row mass
        if conf:
            pr = a / np.clip(a.sum(1, keepdims=True), 1e-30, None)
            wq = np.exp((pr * np.log(pr + 1e-12)).sum(1))  # 1 / n_eff
            b = b * (wq / wq.mean())[:, None]
        m = t.T @ b
        m = m.astype(np.float64)
        m = m + lam * np.linalg.norm(m, 2) * np.eye(len(m))
        try:
            u, _, vt = np.linalg.svd(m)
        except np.linalg.LinAlgError:  # gesdd sporadically fails to converge on finite input here
            u, _, vt = scipy.linalg.svd(m, lapack_driver='gesvd')
        t = nrm(t @ (u @ vt))
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/contrastive_{a.arm}_s{a.seed}.csv', 'w', newline=''))
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
            res = {'gala': loop(x, gal, p)[0], 'gala_own': run(x, gal, p), 'F0.2': loop(xf, gal, p, lam=0.2)[0],
                   'conf': run(x, gal, p, conf=True)}
            for c in (2.0, 4.0, 10.0):
                res[f'ctr{c}'] = run(x, gal, p, c=c)
                res[f'ctrF{c}'] = run(xf, gal, p, lam=0.2, c=c)
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
