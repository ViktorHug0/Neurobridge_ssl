"""Beyond the mean: after q - 0.5*batch mean, also shrink the batch's top-k principal directions by b
(subject/artifact-specific directions shared by the batch's trials). Stream batches of 64.
    centring_pca.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402


def stream(xs, orders, k, b, B=64, a=0.5):
    s = np.concatenate([x[o] for x, o in zip(xs, orders)]); out = np.empty_like(s)
    for i in range(0, len(s), B):
        blk = s[i:i + B]; mu = blk.mean(0); c = blk - mu
        y = blk - a * mu
        if k:
            V = np.linalg.svd(c, full_matrices=False)[2][:k]
            y = y - b * (c @ V.T) @ V
        out[i:i + B] = y
    res, j = [], 0
    for x, o in zip(xs, orders):
        z = np.empty_like(x); z[o] = out[j:j + len(x)]; res.append(z); j += len(x)
    return res


recipes = sys.argv[1:]; data = load(recipes); M = len(recipes); rng = np.random.default_rng(1); res = {}
for kf, subs in FOLDS.items():
    n = len(data[subs[0]]['label']); orders = [rng.permutation(n) for _ in subs]
    for k, b in [(0, 0)] + [(k, b) for k in (1, 2, 4, 8) for b in (0.5, 1.0)]:
        cen = [stream([data[s]['emb'][m] for s in subs], orders, k, b) for m in range(M)]
        for i, s in enumerate(subs):
            res.setdefault(f'centre + PCA k={k} b={b}', {})[s] = score(dict(data[s], emb=[cen[m][i] for m in range(M)]), np.ones((M, n)))
subs = sorted(data); base = np.array([res['centre + PCA k=0 b=0'][s] for s in subs])
print('+'.join(recipes), '(vs centring alone)')
for name, v in res.items():
    v = np.array([v[s] for s in subs])
    print(f'  {name:26s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')
