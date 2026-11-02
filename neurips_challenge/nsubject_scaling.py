"""How does the shuffled-batch transform scale with the number N of participants mixed into the test stream
(sealed N unknown)? Affinity models s1-s4 (REVE-6, 5 training subjects, 15 unseen test subjects each;
affinity_trial_cache.py). The 15 unseen subjects are split into groups of N, each group pooled, shuffled and cut
into batches of 64; centring a=0.5 + shrink of the top-k batch directions (b=0.5).
    nsubject_scaling.py 1 2 3 4"""
import glob, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import score  # noqa: E402


def tf(blk, k, a=.5, b=.5):
    mu = blk.mean(0); c = blk - mu
    if k:
        V = np.linalg.svd(c, full_matrices=False)[2][:k]
        c = c - b * (c @ V.T) @ V
    return c + (1 - a) * mu


rng = np.random.default_rng(5); res = {}
for i in sys.argv[1:]:
    d = glob.glob(f'results/alljoined/track1/affinity/s{i}/reve/2026*')[0]
    z = np.load(f'{d}/trial_dots.npz')
    subs = sorted(int(k[1:].split('_')[0]) for k in z.files if k.endswith('_label'))
    mn = np.mean(np.concatenate([z[f's{s}_norm'] for s in subs]))
    E = {s: dict(emb=[z[f's{s}_emb'].astype(np.float32) / mn], label=z[f's{s}_label'], G=z['gallery']) for s in subs}
    n = len(E[subs[0]]['label'])
    for s in subs:
        res.setdefault('plain', []).append(score(E[s], np.ones((1, n))))
    for N in (1, 3, 5, 15):
        perm = list(rng.permutation(subs))
        for g in range(0, 15, N):
            grp = perm[g:g + N]
            allx = np.concatenate([E[s]['emb'][0] for s in grp]); o = rng.permutation(len(allx))
            for k in (0, 1, 2, 4):
                y = np.empty_like(allx)
                for j in range(0, len(o), 64):
                    y[o[j:j + 64]] = tf(allx[o[j:j + 64]], k)
                for t, s in enumerate(grp):
                    res.setdefault(f'N={N:2d} centre + top-{k} shrink', []).append(score(dict(E[s], emb=[y[t * n:(t + 1) * n]]), np.ones((1, n))))
    print(f'model s{i} done', flush=True)
base = np.array(res['plain'])
order = sorted(res, key=lambda k: (k != 'plain', k))
for name in order:
    v = np.array(res[name]) if name == 'plain' else None
    if name == 'plain':
        print(f'  {name:30s} {v.mean():6.2f}'); continue
    # rows were appended in group order, not subject order: compare means only
    print(f'  {name:30s} {np.mean(res[name]):6.2f}  {np.mean(res[name]) - base.mean():+.2f}')
