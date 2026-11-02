"""Parameter-free fusion from split-half reliabilities: calibrated product of experts.
Each member m gets the inverse temperature beta_m at which its mean max softmax(beta z(S_m)) equals its label-free
accuracy estimate (target), then the log-posteriors are summed (product of experts).
  poe_{set}_{t}   set in {acef = F(0), F(1), S;  path = F(0), F(0.5), F(1), S}; target t: 'sq' = sqrt(a) (half-length
                  accuracy if the halves' errors were independent), 'a' = a
  lin_{set}_{k}   linear a^k weights (as splithalf_fuse2.py) for reference
  dp_{set}_{t}    linear weights = d' of an M-alternative forced choice at accuracy target t (signal detection:
                  optimal linear weight for independent equal-variance Gaussian experts; no free parameter)
usage: splithalf_fuse3.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from agnostic import zrow  # noqa: E402
from path import fuse  # noqa: E402
from splithalf import family  # noqa: E402
from stability import blocks  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def dprime(p, m):
    """Sensitivity d' of an m-alternative forced choice with accuracy p (equal-variance Gaussian model)."""
    x = np.linspace(-8, 14, 4001)
    ds = np.linspace(0, 10, 401)
    acc = np.array([np.trapezoid(norm.pdf(x - d) * norm.cdf(x) ** (m - 1), x) for d in ds])
    return np.interp(np.clip(p, acc[0], acc[-1]), acc, ds)


def calibrated(s, target, lo=1e-2, hi=1e2):
    """log softmax(beta z(s)) with beta such that mean max probability = target (bisection in log beta)."""
    z = zrow(s)
    for _ in range(40):
        b = np.sqrt(lo * hi)
        lo, hi = (b, hi) if softmax(b * z, 1).max(1).mean() < target else (lo, b)
    return log_softmax(np.sqrt(lo * hi) * z, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    ag = pd.read_csv(f'{C}/splithalf_{a.arm}_s{a.seed}.csv').query("metric == 'agree'")
    w = csv.writer(open(f'{C}/shfuse3_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = blocks(a.arm, a.seed, sid)
        q, g, dis, blk = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'blk'))
        halves = (blk[:2].mean(0), blk[2:].mean(0))
        for D in (0, 200, 400, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            af = ag[(ag.subject == sid) & (ag.D == D)].sort_values('r').value.to_numpy() / 100
            F = {r: family(q, gal, p, r) for r in (0.0, 0.5, 1.0)}
            S = score_deploy(q, gal)
            hs = [score_deploy(h, gal).argmax(1) for h in halves]
            a_s = max(float((hs[0] == hs[1]).mean()), 1e-3)
            sets = {'acef': ([F[0.0], F[1.0], S], np.array([af[0], af[10], a_s])),
                    'path': ([F[0.0], F[0.5], F[1.0], S], np.array([af[0], af[5], af[10], a_s]))}
            res = {}
            for name, (mem, am) in sets.items():
                for t, tgt in (('sq', np.sqrt(am)), ('a', am)):
                    res[f'poe_{name}_{t}'] = sum(calibrated(s, tt) for s, tt in zip(mem, tgt))
                for k in (1, 2, 4):
                    res[f'lin_{name}_{k}'] = fuse(mem, am ** k)
                res[f'dp_{name}_sq'] = fuse(mem, dprime(np.sqrt(am), len(gal)))
                res[f'dp_{name}_a'] = fuse(mem, dprime(am, len(gal)))
            for kk, s in res.items():
                w.writerow([sid, D, kk, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
