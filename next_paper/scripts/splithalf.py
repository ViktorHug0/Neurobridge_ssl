"""Split-half reliability of the alignment: does the agreement between two independently aligned repetition halves
identify how much rotation to trust? Each query = 80 repetitions; halves = means of 40 (cache 'blk' = 4 x 20).
For every r of the family F(r) (start (1-r) SAW + r moment matching, Procrustes shrunk with lambda = r; r = 0 GALA):
  top1       F(r) on the 80-rep queries (what we deploy)
  half_acc   mean top-1 of F(r) fitted and scored on each half separately (uses labels: reference only)
  agree      fraction of queries whose two half-alignments pick the same gallery item (label-free; ~ p_half^2
             if the halves' errors are independent)
usage: splithalf.py --seed 3301 [--arm atm128_vc]
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from stability import blocks  # noqa: E402
from path import RS  # noqa: E402
from shared import fit_saw_transform  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def family(q, gal, p, r):
    x = nrm(fit_saw_transform(q, p)[0])
    return loop(nrm((1 - r) * x + r * nrm(mm(q, gal))), gal, p, lam=r)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/splithalf_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'r', 'metric', 'value'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = blocks(a.arm, a.seed, sid)
        q, g, dis, blk = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'blk'))
        halves = (blk[:2].mean(0), blk[2:].mean(0))
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            for r in RS:
                w.writerow([sid, D, r, 'top1', topk_acc(family(q, gal, p, r), y)[1]])
                h = [family(hq, gal, p, r) for hq in halves]
                t = [s.argmax(1) for s in h]
                w.writerow([sid, D, r, 'half_acc', 50 * sum((ti == y).mean() for ti in t)])
                w.writerow([sid, D, r, 'agree', 100 * (t[0] == t[1]).mean()])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
