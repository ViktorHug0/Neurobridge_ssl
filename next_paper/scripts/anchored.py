"""Anchored GALA: one objective whose coupling is a product of two likelihoods, the rotated similarity and a
correspondence-free similarity, instead of fusing separate experts afterwards.
  max_{T, P}  <P, z(S(T)) + w z(S0)> + tau H(P)   P balanced (as GALA),  T orthogonal (Procrustes on P)
S(T) = CSLS of the rotated SAW queries (GALA's score), S0 = moment matching to the gallery + CSLS(10), no rotation
(the transform-free part of SCORE). Final score = z(S(T)) + w z(S0). No D, no N/M.
Methods: gala, m0 (= S0), post (z(G) + z(S0), plain post-hoc fusion), anch{w} (anchored loop, scored by S(T) only),
anchf{w} (anchored loop + same anchor in the final score).
usage: anchored.py --seed 3300 [--arm atm128_vc]
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

WS = (0.5, 1.0, 2.0)


def anchored(t, gal, s0, p, w):
    z0 = zrow(s0)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    for _ in range(max(1, int(p.soft_procrustes_steps))):
        f = zrow(zrow(s) + w * z0) * s.std(1, keepdims=True)  # joint evidence on GALA's score scale
        t = apply_orthogonal_map(t, procrustes(t, gal, damped_assign(f, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), p, 0.0))
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/anchored_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            s0 = score_features(nrm(mm(q, gal)), gal, use_csls=True, csls_k=10)
            G = loop(x, gal, p)[0]
            res = {'gala': G, 'm0': s0, 'post': zrow(G) + zrow(s0)}
            for wt in WS:
                sa = anchored(x, gal, s0, p, wt)
                res[f'anch{wt}'], res[f'anchf{wt}'] = sa, zrow(sa) + wt * zrow(s0)
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
