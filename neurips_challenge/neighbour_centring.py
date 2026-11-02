"""Label-free in-batch routing for shuffled mixed-participant batches: centre each trial on the mean of its K most
similar trials of the batch (similarity of raw-EEG fingerprints: per-channel log-variance + channel correlations,
z-scored with statistics of the batch), instead of the whole-batch mean; then the usual top-direction shrink.
Fold's 5 unseen subjects pooled and shuffled, batches of 64.
    neighbour_centring.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402
from neurips_challenge.subject_separability import eeg_fp  # noqa: E402


def transform(q, f, K, a=.5, b=.5, oracle=None):
    mu_all = q.mean(0)
    if K:
        fz = (f - f.mean(0)) / (f.std(0) + 1e-6)
        if oracle is not None:  # upper bound: centre on the true same-subject trials of the batch
            mu = np.stack([q[oracle == o].mean(0) for o in oracle])
        else:
            D = ((fz[:, None] - fz[None]) ** 2).sum(-1)
            nb = np.argsort(D, 1)[:, :K]                                 # includes the trial itself
            mu = q[nb].mean(1)
    else:
        mu = np.repeat(mu_all[None], len(q), 0)
    c = q - mu
    if b:
        V = np.linalg.svd(c, full_matrices=False)[2][:1]
        c = c - b * (c @ V.T) @ V
    return c + (1 - a) * mu


recipes = sys.argv[1:]; data = load(recipes); M = len(recipes); rng = np.random.default_rng(4); res = {}
cfgs = [('plain', None)] + [(f'K={K}', K) for K in (0, 4, 8, 16, 32)] + [('oracle same-subject groups', 'oracle')]
for kf, subs in FOLDS.items():
    n = len(data[subs[0]]['label'])
    F = np.concatenate([eeg_fp(s) for s in subs]); owner = np.repeat(np.arange(len(subs)), n)
    order = rng.permutation(len(F))
    for name, K in cfgs:
        outs = []
        for m in range(M):
            allx = np.concatenate([data[s]['emb'][m] for s in subs])
            if K is None:
                outs.append(allx); continue
            y = np.empty_like(allx)
            for j in range(0, len(order), 64):
                idx = order[j:j + 64]
                y[idx] = transform(allx[idx], F[idx], 16 if K == 'oracle' else K, oracle=owner[idx] if K == 'oracle' else None)
            outs.append(y)
        for i, s in enumerate(subs):
            res.setdefault(name, {})[s] = score(dict(data[s], emb=[o[i * n:(i + 1) * n] for o in outs]), np.ones((M, n)))
subs = sorted(data); base = np.array([res['plain'][s] for s in subs])
print('+'.join(recipes), '(shuffled mixed 64-batches; K=0 = whole-batch centring)')
for name, v in res.items():
    v = np.array([v[s] for s in subs])
    print(f'  {name:28s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')
