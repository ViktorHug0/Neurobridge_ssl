"""Shuffled mixed-participant batches (sealed condition): sweep the number k of top batch directions shrunk and the
shrink b, the centring a, and combinations with the fully static correction (other subjects' mean + top direction).
Reported per fold pair (screen 1-2 | confirm 3-4), deltas vs plain.
    mixed_sweep.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402


def tf(blk, a, k, b):
    mu = blk.mean(0); c = blk - mu
    if k and b:
        V = np.linalg.svd(c, full_matrices=False)[2][:k]
        c = c - b * (c @ V.T) @ V
    return c + (1 - a) * mu


data = load(sys.argv[1:]); M = len(sys.argv[1:]); rng = np.random.default_rng(3); res = {}
cfgs = [('plain', None, 0, 0, 0, False)]
cfgs += [(f'B=64 a=.5 k={k} b={b}', 64, .5, k, b, False) for k in (1, 2, 3, 4, 8) for b in (.3, .5, .7)]
cfgs += [(f'B=64 a={a} k=1 b=.5', 64, a, 1, .5, False) for a in (.25, .75, 1.0)]
cfgs += [('B=256 a=.5 k=2 b=.5', 256, .5, 2, .5, False), ('B=256 a=.5 k=4 b=.5', 256, .5, 4, .5, False)]
cfgs += [('static mean+dir, then B=64 a=.5 k=1 b=.5', 64, .5, 1, .5, True), ('static mean+dir only', None, 0, 0, 0, True)]
for kf, subs in FOLDS.items():
    n = len(data[subs[0]]['label'])
    outs = {c[0]: [[None] * len(subs) for _ in range(M)] for c in cfgs}
    for m in range(M):
        X = [data[s]['emb'][m] for s in subs]
        stat = []
        for i in range(len(subs)):  # static correction from the OTHER subjects of the fold (same model)
            o = np.concatenate([X[j] for j in range(len(subs)) if j != i]); mu = o.mean(0)
            d = np.linalg.svd(o[rng.choice(len(o), 4000, replace=False)] - mu, full_matrices=False)[2][0]
            c = X[i] - mu; stat.append(c - 0.5 * np.outer(c @ d, d) + 0.5 * mu)
        for name, B, a, k, b, use_static in cfgs:
            src = stat if use_static else X
            allx = np.concatenate(src)
            if B is None:
                y = allx
            else:
                y = np.empty_like(allx); o = rng.permutation(len(allx))
                for j in range(0, len(o), B):
                    y[o[j:j + B]] = tf(allx[o[j:j + B]], a, k, b)
            for i in range(len(subs)):
                outs[name][m][i] = y[i * n:(i + 1) * n]
    for name in outs:
        for i, s in enumerate(subs):
            res.setdefault(name, {})[s] = score(dict(data[s], emb=[outs[name][m][i] for m in range(M)]), np.ones((M, n)))
print('+'.join(sys.argv[1:]), '(shuffled mixed batches; deltas vs plain; folds 1-2 | folds 3-4)')
for name, v in res.items():
    row = []
    for fs in ((1, 2), (3, 4)):
        ss = [s for f in fs for s in FOLDS[f]]
        x = np.array([v[s] for s in ss]); bse = np.array([res['plain'][s] for s in ss])
        row.append(f'{x.mean():6.2f} {x.mean() - bse.mean():+5.2f} ({(x > bse).sum():2d}/{(x < bse).sum():2d})')
    print(f'  {name:42s} ' + ' | '.join(row))
