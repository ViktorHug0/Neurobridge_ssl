"""Reliability-weighted alignment (RWA), frozen after the ATM development runs (splithalf*.py), for held-out encoders.
Members: the trust path F(0) (= GALA), F(0.5), F(1) (= conservative) and the landmark estimator S.
Each member's reliability a_m = top-1 agreement between its two independent alignments of the 40-repetition halves
(own replicates: no independence assumption between members). Score = sum_m a_m^k z(S_m); k = 2 (main, frozen), 1, 4.
Reference: ACEF (cross-expert triplet agreement on F(0), F(1), S) and the members alone. Setting: R3 train distractors.
usage: rwa.py --arm tsconv_vc --seed 3300
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from path import fuse  # noqa: E402
from splithalf import family  # noqa: E402
from stability import blocks  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def members(q, gal, p):
    return {'F0': family(q, gal, p, 0.0), 'F.5': family(q, gal, p, 0.5), 'F1': family(q, gal, p, 1.0),
            'S': score_deploy(q, gal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, default=3300)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/rwa_{a.arm}_s{a.seed}.csv', 'w', newline=''))
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
            h1, h2 = (members(hq, gal, p) for hq in halves)
            am = np.array([max(float((h1[k].argmax(1) == h2[k].argmax(1)).mean()), 1e-3) for k in full])
            mem = list(full.values())
            res = {**full, 'acef': fuse([full['F0'], full['F1'], full['S']], lsq_acc([full['F0'], full['F1'], full['S']])),
                   'sel': mem[int(am.argmax())]}
            for k in (1, 2, 4):
                res[f'rwa{k}'] = fuse(mem, am ** k)
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
            for kk, v in zip(full, am):
                w.writerow([sid, D, f'agree_{kk}', 100 * v])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
