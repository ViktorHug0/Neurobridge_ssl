"""Marginalising the rotation-trust hyperparameter of ONE model family instead of fusing hand-picked experts.
F(r), r in [0, 1]: start = (1-r) SAW + r * moment matching to the gallery, Procrustes shrunk to the identity with
lambda = r (r = 0 is GALA, r = 1 the conservative expert). Score = sum_r w(r) z(S_r), with
  uni      w uniform over the grid
  lsq      w = label-free accuracies from log top-1 agreements, least squares (gated_eval.lsq_acc)
  eig      w = leading eigenvector of the top-1 agreement matrix with its diagonal re-estimated (rank-one fit,
           Parisi et al. 2014 spectral meta-learner)
  *_S      same with the landmark (SCORE-style) scores added as one more member
usage: path.py --seed 3300 [--arm atm128_vc]
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from shared import fit_saw_transform  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402

RS = np.round(np.linspace(0, 1, 11), 2)


def eig_acc(scores, iters=50):
    """Rank-one fit of the off-diagonal top-1 agreement matrix (diagonal re-imputed iteratively)."""
    tops = np.stack([s.argmax(1) for s in scores])
    a = (tops[:, None] == tops[None]).mean(2)
    v = np.sqrt(np.clip(a.mean(1), 1e-3, 1))
    for _ in range(iters):
        np.fill_diagonal(a, v ** 2)
        val, vec = np.linalg.eigh(a)
        v = np.abs(vec[:, -1]) * np.sqrt(max(val[-1], 1e-9))
    return np.clip(v, 1e-3, 1)


def fuse(scores, w):
    return sum(wk * zrow(s) for wk, s in zip(w, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/path_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            mq = nrm(mm(q, gal))
            F = [loop(nrm((1 - r) * x + r * mq), gal, p, lam=r)[0] for r in RS]
            S = score_deploy(q, gal)
            res = {f'r={r}': s for r, s in zip(RS, F)}
            for tag, mem in (('', F), ('_S', F + [S])):
                res[f'uni{tag}'] = fuse(mem, np.ones(len(mem)))
                res[f'lsq{tag}'] = fuse(mem, lsq_acc(mem))
                res[f'eig{tag}'] = fuse(mem, eig_acc(mem))
            res['tri_ends_S'] = fuse([F[0], F[-1], S], lsq_acc([F[0], F[-1], S]))  # = ACEF with G, C, S
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
