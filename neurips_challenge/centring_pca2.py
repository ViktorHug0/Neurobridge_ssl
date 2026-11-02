"""Finer sweep of batch spectral shrinkage after centring, reported per fold pair (screen 1-2 / confirm 3-4).
  pc k b    : q - 0.5*mu - b * projection of (q - mu) on the batch's top-k principal directions
  white g   : partial whitening, singular values s_i of the centred batch -> s_i^(1-g) * mean(s)^g
    centring_pca2.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402


def transform(blk, mode, p1, p2, a=0.5):
    mu = blk.mean(0); c = blk - mu
    if mode == 'none':
        return blk - a * mu
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    if mode == 'pc':
        S2 = S.copy(); S2[:p1] *= (1 - p2)
    else:  # white
        S2 = S ** (1 - p1) * S.mean() ** p1
    return (U * S2) @ Vt + (1 - a) * mu


def stream(xs, orders, mode, p1, p2, B):
    s = np.concatenate([x[o] for x, o in zip(xs, orders)]); out = np.empty_like(s)
    for i in range(0, len(s), B):
        out[i:i + B] = transform(s[i:i + B], mode, p1, p2)
    res, j = [], 0
    for x, o in zip(xs, orders):
        z = np.empty_like(x); z[o] = out[j:j + len(x)]; res.append(z); j += len(x)
    return res


recipes = sys.argv[1:]; data = load(recipes); M = len(recipes); rng = np.random.default_rng(1); res = {}
cfgs = [('none', 0, 0, 64)] + [('pc', k, b, 64) for k in (1, 2, 3) for b in (0.25, 0.4, 0.5, 0.6, 0.75)] \
     + [('white', g, 0, 64) for g in (0.25, 0.5, 0.75)] + [('pc', 1, 0.5, B) for B in (32, 128, 256)] + [('pc', 2, 0.5, B) for B in (32, 128, 256)]
for kf, subs in FOLDS.items():
    n = len(data[subs[0]]['label']); orders = [rng.permutation(n) for _ in subs]
    for mode, p1, p2, B in cfgs:
        cen = [stream([data[s]['emb'][m] for s in subs], orders, mode, p1, p2, B) for m in range(M)]
        for i, s in enumerate(subs):
            res.setdefault(f'{mode} {p1} {p2} B={B}', {})[s] = score(dict(data[s], emb=[cen[m][i] for m in range(M)]), np.ones((M, n)))
    for s in subs:
        res.setdefault('plain (no centring)', {})[s] = score(data[s], np.ones((M, n)))
base_key = 'none 0 0 B=64'
print('+'.join(recipes), '(deltas vs centring alone; folds 1-2 | folds 3-4)')
for name, v in res.items():
    out = []
    for fs in ((1, 2), (3, 4)):
        ss = [s for f in fs for s in FOLDS[f]]
        x = np.array([v[s] for s in ss]); b = np.array([res[base_key][s] for s in ss])
        out.append(f'{x.mean():6.2f} {x.mean() - b.mean():+5.2f} ({(x > b).sum():2d}/{(x < b).sum():2d})')
    print(f'  {name:22s} ' + ' | '.join(out))
