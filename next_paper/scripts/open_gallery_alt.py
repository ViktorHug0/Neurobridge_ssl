"""Alternating support refinement for R3: robust GALA (blend start + ratio shrinkage), then K rounds of
[support = rectangular Hungarian on the current scores -> GALA fitted on the support from the robust start -> rescore
on the full gallery]. Reports support precision and top-1 per round (round 0 = robust GALA alone).
usage: open_gallery_alt.py --seed 3301 [--rounds 4]
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--rounds', type=int, default=4)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/alt_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'round', 'support_precision', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = z['q'].astype(np.float32), z['g'].astype(np.float32), z['d'].astype(np.float32)
        for D in [d for d in D_LIST if d > 0]:
            gal = np.concatenate([g, dis[:D]], 0)
            n, m = len(q), len(gal)
            r = 1 - n / m
            start = nrm((1 - r) * nrm(fit_saw_transform(q, p)[0]) + r * nrm(mm(q, gal)))
            s = loop(start, gal, p, lam=r)[0]
            w.writerow([sid, D, 0, np.nan, topk_acc(s, y)[1]])
            for k in range(1, a.rounds + 1):
                cols = np.sort(linear_sum_assignment(-s)[1])
                t = loop(start, gal[cols], p)[1]
                s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
                w.writerow([sid, D, k, 100 * (cols < 200).mean(), topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
