"""Plain vs AdaBN (test-time batch statistics in every BatchNorm) caches, each with and without stream batch-mean
centring (B=64, a=0.5).   adabn_compare.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402
from neurips_challenge.centring_stream import centre_stream  # noqa: E402

recipes = sys.argv[1:]; M = len(recipes); rng = np.random.default_rng(1); res = {}
for tag, fname in (('plain', 'trial_dots.npz'), ('adabn', 'trial_dots_adabn.npz')):
    data = load(recipes, fname)
    for k, subs in FOLDS.items():
        n = len(data[subs[0]]['label']); order = [rng.permutation(n) for _ in subs]
        cen = [centre_stream([data[s]['emb'][m] for s in subs], order, 64, 0.5) for m in range(M)]
        for i, s in enumerate(subs):
            res.setdefault(tag, {})[s] = score(data[s], np.ones((M, n)))
            res.setdefault(tag + ' + centring', {})[s] = score(dict(data[s], emb=[cen[m][i] for m in range(M)]), np.ones((M, n)))
subs = sorted(res['plain']); base = np.array([res['plain'][s] for s in subs])
print('+'.join(recipes))
for name, v in res.items():
    v = np.array([v[s] for s in subs])
    print(f'  {name:20s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')
