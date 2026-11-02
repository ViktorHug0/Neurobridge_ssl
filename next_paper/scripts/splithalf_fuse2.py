"""Every member weighted by its OWN split-half reliability (no cross-expert independence assumption).
Members: F(r) (trust path, agreements from splithalf.py's csv) and the landmark estimator S (agreement computed here
on the same two 40-repetition halves).
  m{k}        sum over {F(r), r in RS} + {S} of a^k z(.)                       (11 + 1 members)
  c{k}        same on the coarse path {F(0), F(0.5), F(1)} + {S}
  b{k}        two blocks: path average A = sum_r a_r^k z(F_r) / sum_r a_r^k with reliability abar = sum a^(k+1) / sum a^k,
              then abar^k z(A) + a_S^k z(S)
  sb          m1 with Spearman-Brown full-length reliabilities 2a / (1 + a) as weights
usage: splithalf_fuse2.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from agnostic import zrow  # noqa: E402
from path import RS, fuse  # noqa: E402
from splithalf import family  # noqa: E402
from stability import blocks  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    ag = pd.read_csv(f'{C}/splithalf_{a.arm}_s{a.seed}.csv').query("metric == 'agree'")
    w = csv.writer(open(f'{C}/shfuse2_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    coarse = [0, 5, 10]
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = blocks(a.arm, a.seed, sid)
        q, g, dis, blk = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'blk'))
        halves = (blk[:2].mean(0), blk[2:].mean(0))
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            af = ag[(ag.subject == sid) & (ag.D == D)].sort_values('r').value.to_numpy() / 100
            F = [family(q, gal, p, r) for r in RS]
            S = score_deploy(q, gal)
            hs = [score_deploy(h, gal).argmax(1) for h in halves]
            a_s = max(float((hs[0] == hs[1]).mean()), 1e-3)
            mem, am = F + [S], np.append(af, a_s)
            res = {'sb': fuse(mem, 2 * am / (1 + am))}
            for k in (1, 2, 4):
                res[f'm{k}'] = fuse(mem, am ** k)
                res[f'c{k}'] = fuse([F[i] for i in coarse] + [S], np.append(af[coarse], a_s) ** k)
                A = fuse(F, af ** k / (af ** k).sum())
                abar = (af ** (k + 1)).sum() / (af ** k).sum()
                res[f'b{k}'] = abar ** k * zrow(A) + a_s ** k * zrow(S)
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
            w.writerow([sid, D, 'agree_S', 100 * a_s])
            w.writerow([sid, D, 'agree_Fmax', 100 * af.max()])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
