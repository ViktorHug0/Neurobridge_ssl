"""Candidate final predict()-time post-processing under the sealed conditions (shuffled batches mixing
participants, no state): label-free neighbour routing + top-k shrink.
  whole   : centre on the batch mean (a=.5), shrink the top-k centred directions (b=.5)
  route K : centre each trial on the mean of its K nearest batch-mates in raw-EEG fingerprint space (fingerprint
            z-scored with the batch's own statistics), then the same shrink
Mode 'cv': the 4 cv folds, ensemble or single members (5 unseen subjects pooled per fold).
Mode 'aff': affinity models s1-s4, 15 unseen subjects split into groups of N (N-scaling).
    route_final.py cv <recipe> [<recipe> ...]
    route_final.py aff 1 2 3 4"""
import glob, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402
from neurips_challenge.neighbour_lda import fingerprint  # noqa: E402


def post(q, f, K, k=2, a=.5, b=.5):
    if K:
        fz = (f - f.mean(0)) / (f.std(0) + 1e-6)
        D = ((fz[:, None] - fz[None]) ** 2).sum(-1)
        mu = q[np.argsort(D, 1)[:, :K]].mean(1)
    else:
        mu = np.repeat(q.mean(0)[None], len(q), 0)
    c = q - mu
    V = np.linalg.svd(c, full_matrices=False)[2][:k]
    return c - b * (c @ V.T) @ V + (1 - a) * mu


def stream(members, F, rng, cfgs):
    """members: list over models of (n_total, D); F: (n_total, f). Returns {cfg: list over models of outputs}."""
    o = rng.permutation(len(F)); out = {}
    for name, K, k in cfgs:
        ys = []
        for x in members:
            y = np.empty_like(x)
            for j in range(0, len(o), 64):
                idx = o[j:j + 64]; y[idx] = post(x[idx], F[idx], K, k)
            ys.append(y)
        out[name] = ys
    return out


CFGS = [('whole k=1', 0, 1), ('whole k=2', 0, 2), ('route K=8 k=1', 8, 1), ('route K=8 k=2', 8, 2), ('route K=4 k=2', 4, 2)]
rng = np.random.default_rng(8); res = {}
FP = {s: fingerprint(s) for s in range(1, 21)}
if sys.argv[1] == 'cv':
    recipes = sys.argv[2:]; data = load(recipes); M = len(recipes)
    for kf, subs in FOLDS.items():
        n = len(data[subs[0]]['label'])
        out = stream([np.concatenate([data[s]['emb'][m] for s in subs]) for m in range(M)], np.concatenate([FP[s] for s in subs]), rng, CFGS)
        for name, ys in out.items():
            for i, s in enumerate(subs):
                res.setdefault(name, {})[s] = score(dict(data[s], emb=[y[i * n:(i + 1) * n] for y in ys]), np.ones((M, n)))
        for s in subs:
            res.setdefault('plain', {})[s] = score(data[s], np.ones((M, n)))
    subs = sorted(data); base = np.array([res['plain'][s] for s in subs])
    print('+'.join(recipes), '(cv, shuffled mixed 64-batches of 5 unseen subjects; folds 1-2 | 3-4)')
    for name, v in res.items():
        row = []
        for fs in ((1, 2), (3, 4)):
            ss = [s for f in fs for s in FOLDS[f]]; x = np.array([v[s] for s in ss]); bse = np.array([res['plain'][s] for s in ss])
            row.append(f'{x.mean():6.2f} {x.mean() - bse.mean():+5.2f} ({(x > bse).sum():2d}/{(x < bse).sum():2d})')
        print(f'  {name:16s} ' + ' | '.join(row))
else:
    for i in sys.argv[2:]:
        d = glob.glob(f'results/alljoined/track1/affinity/s{i}/reve/2026*')[0]; z = np.load(f'{d}/trial_dots.npz')
        subs = sorted(int(k[1:].split('_')[0]) for k in z.files if k.endswith('_label'))
        mn = np.mean(np.concatenate([z[f's{s}_norm'] for s in subs]))
        E = {s: dict(emb=[z[f's{s}_emb'].astype(np.float32) / mn], label=z[f's{s}_label'], G=z['gallery']) for s in subs}
        n = len(E[subs[0]]['label'])
        for s in subs:
            res.setdefault('plain', []).append(score(E[s], np.ones((1, n))))
        for N in (5, 15):
            perm = list(rng.permutation(subs))
            for g in range(0, 15, N):
                grp = perm[g:g + N]
                out = stream([np.concatenate([E[s]['emb'][0] for s in grp])], np.concatenate([FP[s] for s in grp]), rng, CFGS)
                for name, ys in out.items():
                    for t, s in enumerate(grp):
                        res.setdefault(f'N={N:2d} {name}', []).append(score(dict(E[s], emb=[ys[0][t * n:(t + 1) * n]]), np.ones((1, n))))
        print(f'model s{i} done', flush=True)
    base = np.mean(res['plain']); print(f'  plain {base:.2f}')
    for name, v in res.items():
        if name != 'plain':
            print(f'  {name:24s} {np.mean(v):6.2f}  {np.mean(v) - base:+.2f}')
