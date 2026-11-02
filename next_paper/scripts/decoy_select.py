"""Decoy-based label-free selection for open galleries (needs no knowledge of D or of the distractor share).
Candidates = start {saw, half, mm} x shrinkage lambda {0, 0.1, 0.3, 1, 3} (GALA loop, balanced Sinkhorn).
Decoy-hit rate of a candidate = fraction of queries whose best CSLS match, in gallery + decoys, is a decoy
(n_dec decoys = known-absent images from the distractor pool, never in the retrieval gallery).
Per subject and D: the candidate with the lowest decoy-hit rate is selected (ties -> smaller lambda, then saw).
Rows: every candidate's top-1 and decoy-hit rate, so selection rules can be compared offline.
usage: decoy_select.py --seed 3301 [--n_dec 500]
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402

LAMS = (0.0, 0.1, 0.3, 1.0, 3.0)
STARTS = ('saw', 'half', 'mm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n_dec', type=int, default=500)
    ap.add_argument('--in_fit', action='store_true', help='decoys are mixed into the gallery during the fit (knockoff-style)')
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    tag = f'decoyfit{a.n_dec}' if a.in_fit else 'decoy'
    w = csv.writer(open(f'{C}/{tag}_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'start', 'lam', 'top1', 'decoy_hit'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis, dec = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'dec'))
        dec = dec[:a.n_dec]
        saw = nrm(fit_saw_transform(q, p)[0])
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            fit_gal = np.concatenate([gal, dec], 0) if a.in_fit else gal
            mmq = nrm(mm(q, fit_gal))
            starts = {'saw': saw, 'half': nrm(saw + mmq), 'mm': mmq}
            for sn, st in starts.items():
                for lam in LAMS:
                    t = loop(st, fit_gal, p, lam=lam)[1]
                    top1 = topk_acc(score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k), y)[1]
                    sd = score_features(t, np.concatenate([gal, dec], 0), use_csls=p.use_csls, csls_k=p.csls_k)
                    w.writerow([sid, D, sn, lam, top1, 100 * (sd.argmax(1) >= len(gal)).mean()])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
