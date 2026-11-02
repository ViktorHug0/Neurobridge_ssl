"""Expert fusion for open galleries, D-agnostic. Experts (all computed on the same gallery, no knowledge of D):
  G   GALA (SAW start, no shrinkage)
  C1  conservative: moment-matching start, Procrustes shrunk with lambda = 1
  C3  conservative: moment-matching start, lambda = 3
  M   moment matching + CSLS, no rotation
Fusions of row-z-scored scores: fixed weights, and per-query confidence weights (each expert weighted by its own
top-1 minus top-2 z-margin). 'inloop' = GALA whose Sinkhorn plan is computed on the fused scores G + w*C1 at every
step (the conservative expert is fixed; GALA's rotation is refitted from the fused plan).
usage: agnostic2.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm, procrustes  # noqa: E402
from agnostic import zrow  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def margin(z):
    t = np.sort(z, 1)
    return (t[:, -1] - t[:, -2])[:, None]


def inloop(t, gal, p, cons, w):
    zc = zrow(cons)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    for _ in range(int(p.soft_procrustes_steps)):
        f = (zrow(s) + w * zc) / (1 + w)
        a = damped_assign(f * s.std(1, keepdims=True), p.sinkhorn_tau, p.sinkhorn_iters, 1.0)  # back to score scale
        t = nrm(t @ procrustes(t, gal, a, p, 0.0))
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return zrow(s) + w * zc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/agnostic2_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        q_saw = nrm(fit_saw_transform(q, p)[0])
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            mmq = nrm(mm(q, gal))
            E = {'G': loop(q_saw, gal, p)[0], 'C1': loop(mmq, gal, p, lam=1.0)[0], 'C3': loop(mmq, gal, p, lam=3.0)[0],
                 'M': score_features(mmq, gal, use_csls=True, csls_k=10)}
            Z = {k: zrow(v) for k, v in E.items()}
            res = {k: v for k, v in E.items()}
            for wt in (2, 3, 4, 6, 10):
                res[f'G+{wt}C1'] = Z['G'] + wt * Z['C1']
            for wt in (2, 4):
                res[f'G+{wt}C3'] = Z['G'] + wt * Z['C3']
                res[f'G+{wt}M'] = Z['G'] + wt * Z['M']
            res['conf_G_C1'] = margin(Z['G']) * Z['G'] + margin(Z['C1']) * Z['C1']
            res['conf_G_C1_M'] = sum(margin(Z[k]) * Z[k] for k in ('G', 'C1', 'M'))
            for wt in (1, 2):
                res[f'inloop{wt}'] = inloop(q_saw, gal, p, E['C1'], wt)
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
