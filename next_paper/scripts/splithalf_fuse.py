"""Split-half-weighted marginalisation of the trust level r (agreements from splithalf.py's csv), vs selection and ACEF.
  sel        F(r*) with r* = argmax_r split-half agreement
  wk         sum_r agree_r^k z(F_r), k = 1, 2, 4
  sel_S      z(F_r*) + z(S)            (S = landmark expert)
  acef       ACEF on {F(0), F(1), S}
usage: splithalf_fuse.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from path import RS, fuse  # noqa: E402
from splithalf import family  # noqa: E402
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
    w = csv.writer(open(f'{C}/shfuse_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            agr = ag[(ag.subject == sid) & (ag.D == D)].sort_values('r').value.to_numpy() / 100
            F = [family(q, gal, p, r) for r in RS]
            S = score_deploy(q, gal)
            i = int(agr.argmax())
            res = {'sel': F[i], 'sel_S': zrow(F[i]) + zrow(S), 'CS': zrow(F[-1]) + zrow(S),
                   'acef': fuse([F[0], F[-1], S], lsq_acc([F[0], F[-1], S]))}
            for k in (1, 2, 4, 8):
                res[f'w{k}'] = fuse(F, agr ** k)
                res[f'w{k}_S'] = fuse(F, agr ** k) / (agr ** k).sum() + zrow(S)
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
