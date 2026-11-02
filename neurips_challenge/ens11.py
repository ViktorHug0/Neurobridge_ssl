"""Ensemble member selection UNDER the batch transform, N = 11 composite testbed (n11.py). The per-member transform
is followed by a sum over members and a sum over repeats, both linear, so each member's per-(subject, image) sums
Q_m are cached once per (config, draw) and any subset scores as sum_m Q_m. Subsets are chosen on one subject half
(folds 1-2 or 3-4) and reported on the other.
    ens11.py <recipe> [<recipe> ...]
"""
import itertools, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.n11 import S, T, static_fit  # noqa: E402
from neurips_challenge.trial_routing import FOLDS, load  # noqa: E402

CFGS = {'plain': None, 'E25': T(), 'k4': T(k=4), 'static k3 r2 c.5': S(k=3, r=2, c=.5)}
recipes = sys.argv[1:]
draws = [sorted(np.random.default_rng(11 + d).choice(np.arange(1, 21), 11, replace=False).tolist()) for d in range(4)]
Q, G = {}, None   # Q[cfg][m][d]: (11, 200, D)
for m, r in enumerate(recipes):
    data = load([r])
    G = torch.tensor(next(iter(data.values()))['G'], dtype=torch.float32)
    for d, subs in enumerate(draws):
        x = torch.tensor(np.concatenate([data[s]['emb'][0] for s in subs]))
        key = torch.tensor(np.concatenate([i * 200 + data[s]['label'] for i, s in enumerate(subs)]))
        o = torch.tensor(np.random.default_rng(100 + d).permutation(len(key))); nb = len(o) // 64; o = o[:nb * 64]
        xb = x[o].view(nb, 64, -1); xb = xb / xb.norm(dim=-1).mean(1)[:, None, None]
        ctx = static_fit(data, 1, [s for s in data if s not in subs])
        for name, f in CFGS.items():
            y = (xb if f is None else f(xb, 0, ctx)).reshape(nb * 64, -1)
            Q.setdefault(name, {}).setdefault(m, {})[d] = torch.zeros(len(subs) * 200, y.shape[1]).index_add_(0, key[o], y).view(len(subs), 200, -1)
    print(f'{r} cached', flush=True); del data

HALF = {1: {s for f in (1, 2) for s in FOLDS[f]}, 2: {s for f in (3, 4) for s in FOLDS[f]}}


def score(name, members):
    """{(subject, draw): top5}"""
    out = {}
    for d, subs in enumerate(draws):
        Sm = sum(Q[name][m][d] for m in members) @ G.T
        t5 = 100 * ((Sm > Sm.diagonal(dim1=1, dim2=2)[..., None]).sum(-1) < 5).float().mean(1)
        out.update({(s, d): t5[i].item() for i, s in enumerate(subs)})
    return out


def mean(v, h):
    return np.mean([x for (s, _), x in v.items() if s in HALF[h]])


for name in CFGS:
    print(f'== {name}')
    subsets = [c for n in (1, 2, 3, 4) for c in itertools.combinations(range(len(recipes)), n)]
    sc = {c: score(name, c) for c in subsets}
    ref = tuple(recipes.index(r) for r in ('reve_scratch_things', 'sqf_dbn_things', 'tsc_dbn40_things') if r in recipes)
    if len(ref) == 3:
        print(f'  current ens  {mean(sc[ref], 1):6.2f} | {mean(sc[ref], 2):6.2f}')
    for h in (1, 2):
        for n in (1, 2, 3, 4):
            best = max((c for c in subsets if len(c) == n), key=lambda c: mean(sc[c], h))
            print(f'  size {n} picked on half {h}: {"+".join(recipes[i] for i in best):70s} pick {mean(sc[best], h):6.2f}  other half {mean(sc[best], 3 - h):6.2f}')
