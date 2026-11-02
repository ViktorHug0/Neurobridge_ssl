"""Ensemble search with the batch transform (stream batches of 64, random order within subject, subjects back to
back): each member's output q -> (q - mu) - b * proj_top-k(q - mu) + (1 - a) * mu, members then summed.
    ens_transform.py reduce <recipe> ...     per-(subject, image) means, plain and transformed -> routing/Q_<recipe>.npz
    ens_transform.py search [--max 4] <recipe> ...
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load  # noqa: E402

OUT = 'results/alljoined/track1/routing'
A, K, BSHRINK, B = 0.5, 1, 0.5, 64


def transform(blk, a=A, k=K, b=BSHRINK):
    mu = blk.mean(0); c = blk - mu
    V = np.linalg.svd(c, full_matrices=False)[2][:k]
    return c - b * (c @ V.T) @ V + (1 - a) * mu


def reduce(recipe):
    data = load([recipe]); rng = np.random.default_rng(1); Q = {}
    for kf, subs in FOLDS.items():
        n = len(data[subs[0]]['label']); orders = [rng.permutation(n) for _ in subs]
        s_all = np.concatenate([data[s]['emb'][0][o] for s, o in zip(subs, orders)]); out = np.empty_like(s_all)
        for i in range(0, len(s_all), B):
            out[i:i + B] = transform(s_all[i:i + B])
        j = 0
        for s, o in zip(subs, orders):
            e = data[s]; y = np.empty_like(e['emb'][0]); y[o] = out[j:j + n]; j += n
            for tag, x in (('plain', e['emb'][0]), ('tf', y)):
                q = np.zeros((200, x.shape[1])); np.add.at(q, e['label'], x); Q[f'{tag}_s{s}'] = q / np.bincount(e['label'], minlength=200)[:, None]
            Q['gallery'] = e['G']
    np.savez(f'{OUT}/Q_{recipe}.npz', **Q)
    print('reduced', recipe, flush=True)


def top5(Q, G):
    S = Q @ G.T
    return 100 * np.mean([(S[i] > S[i, i]).sum() < 5 for i in range(len(S))])


def search(recipes, max_size):
    Z = {r: np.load(f'{OUT}/Q_{r}.npz') for r in recipes}; G = Z[recipes[0]]['gallery']
    subs = sorted(s for f in FOLDS.values() for s in f)
    def acc(members, tag):
        return np.array([top5(sum(Z[r][f'{tag}_s{s}'] for r in members), G) for s in subs])
    for tag in ('plain', 'tf'):
        print(f'== {tag}')
        singles = {r: acc([r], tag) for r in recipes}
        for r, v in sorted(singles.items(), key=lambda t: -t[1].mean()):
            print(f'  {r:22s} {v.mean():6.2f}')
        for k in range(2, max_size + 1):
            best = max(itertools.combinations(recipes, k), key=lambda c: acc(c, tag).mean())
            print(f'  best {k}: {" + ".join(best)}  {acc(best, tag).mean():.2f}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'reduce':
        for r in args:
            reduce(r)
    else:
        m = 4
        if args[0] == '--max':
            m, args = int(args[1]), args[2:]
        search(args, m)
