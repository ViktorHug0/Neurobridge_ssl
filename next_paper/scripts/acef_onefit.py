"""Close the gap to the correspondence bound at large D: ONE rotation fitted from ACEF's fused plan (no further GALA loop,
so the fit is not re-attached to distractors), pair weights = plan^gamma (sharpening = soft trimming), then scored on
the full gallery. Variants:
  R_soft{g}   weighted Procrustes on the fused Sinkhorn plan ** g (row-normalised plan, balanced columns as GALA)
  R_hung{g}   Procrustes on ACEF's Hungarian pairs weighted by their fused z-margin ** g
  acef4_*     ACEF re-fused with R_* as a 4th expert (accuracies by least squares on log-agreements)
Also the bound (true pairs among ACEF's Hungarian matches). ATM, D in {0, 200, 800, 1654}.
usage: acef_onefit.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def fuse(experts):
    return sum(w * zrow(s) for w, s in zip(lsq_acc(list(experts)), experts))


def rot(x, y, w):
    u, _, vt = np.linalg.svd((x * w[:, None]).T @ y)
    return u @ vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/onefit_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        y = np.arange(200)
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            gn = nrm(gal)
            G = loop(x, gal, p)[0]
            Cc = loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0]
            L = score_deploy(q, gal)
            f = fuse([G, Cc, L])
            res = {'acef': f}
            sc = lambda r: score_features(nrm(x @ r), gal, use_csls=p.use_csls, csls_k=p.csls_k)
            plan = damped_assign(zrow(f) * float(G.std(1).mean()), p.sinkhorn_tau, p.sinkhorn_iters, 1.0)
            plan = plan / plan.sum(1, keepdims=True)
            cols = linear_sum_assignment(-f)[1]
            zf = zrow(f)
            srt = np.sort(zf, 1)
            margin = np.clip(zf[y, cols] - np.where(srt[:, -1] == zf[y, cols], srt[:, -2], srt[:, -1]), 1e-6, None)
            for gm in (1, 2, 4, 8):
                pw = plan ** gm
                c = x.T @ pw @ gn
                u, _, vt = np.linalg.svd(c)
                res[f'R_soft{gm}'] = sc(u @ vt)
                res[f'R_hung{gm}'] = sc(rot(x, gn[cols], margin ** gm))
            ok = cols == y
            res['bound'] = sc(rot(x[ok], gn[cols][ok], np.ones(ok.sum())))
            for k in [k for k in list(res) if k.startswith('R_')]:
                res[f'acef4_{k}'] = fuse([G, Cc, L, res[k]])
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
