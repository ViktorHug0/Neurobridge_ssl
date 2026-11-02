"""Robust GALA with an ESTIMATED absent share (no knowledge of D, N/M or coverage).
1. Column statistic, computed before any fit (so decoys and distractors stay exchangeable): c_j = max_i CSLS(x_i, v_j)
   over the pool gallery + decoys, x = queries after SAW whitening.
2. Empirical p-value of each gallery column against the decoy statistics (decoys = known-absent images from the
   distractor pool, never retrieved), Storey's estimator pi0 = #{p > l} / ((1 - l) M), clipped to [0, 1].
3. Robust GALA with r = pi0: start (1 - r) SAW + r * moment matching to the gallery, Procrustes shrunk by lambda = r.
Rows: true r = D / (200 + D), pi0 per Storey lambda, top-1 of robust GALA with r_hat and with the true r.
usage: pi0_gala.py --seed 3301 [--n_dec 500]
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def pi0(stat_gal, stat_dec, lam):
    p = (1 + (stat_dec[None, :] >= stat_gal[:, None]).sum(1)) / (1 + len(stat_dec))
    return float(np.clip((p > lam).sum() / ((1 - lam) * len(p)), 0, 1))


def robust(q_saw, q, gal, p, r):
    start = nrm((1 - r) * q_saw + r * nrm(mm(q, gal)))
    return loop(start, gal, p, lam=r)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n_dec', type=int, default=500)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/pi0_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'stat', 'r_true', 'auc', 'pi0', 'top1_hat', 'top1_true', 'top1_gala'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis, dec = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'dec'))
        dec = dec[:a.n_dec]
        q_saw = nrm(fit_saw_transform(q, p)[0])
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            pool = np.concatenate([gal, dec], 0)
            rt = D / (200 + D)
            acc = lambda sc: topk_acc(sc, y)[1]
            top_true, top_gala = acc(robust(q_saw, q, gal, p, rt)), acc(loop(q_saw, gal, p)[0])
            stats = {'start': score_features(q_saw, pool, use_csls=p.use_csls, csls_k=p.csls_k)}
            for pr in (0.0, 0.5):  # pilot = robust GALA with a fixed r on gallery + decoys (decoys stay exchangeable)
                stats[f'pilot{pr}'] = robust(q_saw, q, pool, p, pr)
            for name, sc in stats.items():
                col = sc.max(0)
                cg, cd = col[:len(gal)], col[len(gal):]
                lab = np.r_[np.ones(200), np.zeros(len(gal) - 200 + len(cd))]
                allc = np.r_[cg[:200], cg[200:], cd]
                auc = float((allc[lab == 1][:, None] > allc[lab == 0][None, :]).mean())
                r5 = pi0(cg, cd, 0.5)
                w.writerow([sid, D, name, rt, auc, r5, acc(robust(q_saw, q, gal, p, r5)), top_true, top_gala])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
