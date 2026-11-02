"""D-agnostic candidates for open galleries (no knowledge of D, N/M or coverage).
  fuse{w}     z-scored GALA scores + w * z-scored conservative scores (moment-matching start, lambda = 1), per query
  stab        rotation instability: GALA refitted on B = 5 random 80% subsets of the gallery columns; mean pairwise
              ||R_b - R_c||_F / sqrt(2d) (diagnostic; also used as lambda in stabshr)
  stabshr     robust GALA with r = instability (start blend and lambda), no other knob
usage: agnostic.py --seed 3301
"""
import argparse, csv, itertools, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm, procrustes  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def zrow(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def rotation(t, gal, p):
    """Cumulative GALA rotation (balanced plan, no shrinkage)."""
    rot = np.eye(t.shape[1], dtype=np.float32)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    for _ in range(int(p.soft_procrustes_steps)):
        m = procrustes(t, gal, damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0), p, 0.0)
        t, rot = nrm(t @ m), rot @ m
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/agnostic_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'value'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        q_saw = nrm(fit_saw_transform(q, p)[0])
        rng = np.random.default_rng(sid)
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            s_g = loop(q_saw, gal, p)[0]
            s_c = loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0]
            for wt in (0.5, 1.0, 2.0):
                w.writerow([sid, D, f'fuse{wt}', topk_acc(zrow(s_g) + wt * zrow(s_c), y)[1]])
            rots = [rotation(q_saw, gal[np.sort(rng.choice(len(gal), int(0.8 * len(gal)), replace=False))], p)
                    for _ in range(5)]
            inst = float(np.mean([np.linalg.norm(r1 - r2) / np.sqrt(2 * len(r1)) for r1, r2 in itertools.combinations(rots, 2)]))
            w.writerow([sid, D, 'stab', inst])
            st = nrm((1 - inst) * q_saw + inst * nrm(mm(q, gal)))
            w.writerow([sid, D, 'stabshr', topk_acc(loop(st, gal, p, lam=inst)[0], y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
