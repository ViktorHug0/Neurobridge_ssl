"""Maximal-reproducibility fusion: one objective, no exponent.
  w* = argmax_{w in simplex} agree( argmax_j sum_m w_m z(S_m^half1), argmax_j sum_m w_m z(S_m^half2) )
over members {F(0), F(0.5), F(1), S} (simplex grid, step 0.1; ties -> mean of the tied w), then score the 80-rep
queries with sum_m w*_m z(S_m). Reference: rwa2 (own-agreement^2 weights) and ACEF.
usage: rwa_opt.py --arm atm128_vc --seed 3300
"""
import argparse, csv, itertools, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from path import fuse  # noqa: E402
from rwa import members  # noqa: E402
from stability import blocks  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402

GRID = np.array([w for w in itertools.product(range(11), repeat=4) if sum(w) == 10]) / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, default=3300)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/rwaopt_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = blocks(a.arm, a.seed, sid)
        q, g, dis, blk = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'blk'))
        halves = (blk[:2].mean(0), blk[2:].mean(0))
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            full = members(q, gal, p)
            zh = [np.stack([zrow(s) for s in members(hq, gal, p).values()]) for hq in halves]
            agr = np.array([(np.tensordot(wt, zh[0], 1).argmax(1) == np.tensordot(wt, zh[1], 1).argmax(1)).mean() for wt in GRID])
            wopt = GRID[agr == agr.max()].mean(0)
            own = np.array([(zh[0][m].argmax(1) == zh[1][m].argmax(1)).mean() for m in range(4)])
            mem = list(full.values())
            res = {'opt': fuse(mem, wopt), 'rwa2': fuse(mem, own ** 2),
                   'acef': fuse([full['F0'], full['F1'], full['S']], lsq_acc([full['F0'], full['F1'], full['S']]))}
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
            for kk, v in zip(full, wopt):
                w.writerow([sid, D, f'w_{kk}', v])
            w.writerow([sid, D, 'agree_opt', 100 * agr.max()])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
