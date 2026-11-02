"""Held-out check of the simplest upgrades: step shrinkage only (shr{l}: SAW start, every Procrustes step shrunk to the
identity with lambda l) vs the blended start F{r} (start (1-r) SAW + r moment matching, lambda r). R3 train distractors.
usage: simple2.py --arm tsconv_vc [--seed 3300]
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, default=3300)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/simple2_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            res = {'gala': loop(x, gal, p)[0]}
            for lam in (0.1, 0.2, 0.3):
                res[f'shr{lam}'] = loop(x, gal, p, lam=lam)[0]
            for r in (0.1, 0.2):
                res[f'F{r}'] = loop(nrm((1 - r) * x + r * nrm(mm(q, gal))), gal, p, lam=r)[0]
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
