"""The sealed grader SHUFFLES test batches across participants (organisers, Discord 2026-10-09: batch statistics
allowed, no state across predict calls, batches may mix participants, batch size may change). Re-test the batch
transform when each batch of B is drawn uniformly from all trials of the fold's 5 test subjects pooled, and
compare to static (batch-independent) corrections fitted on OTHER subjects' outputs.
    mixed_batch.py <recipe> [<recipe> ...]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402


def tf(blk, a, b, k=1, static_dir=None):
    mu = blk.mean(0); c = blk - mu
    if b:
        V = static_dir[None] if static_dir is not None else np.linalg.svd(c, full_matrices=False)[2][:k]
        c = c - b * (c @ V.T) @ V
    return c + (1 - a) * mu


def main(recipes):
    data = load(recipes); M = len(recipes); rng = np.random.default_rng(2); res = {}
    for kf, subs in FOLDS.items():
        n = len(data[subs[0]]['label'])
        for m in range(M):
            pass
        cfgs = [('plain', None, 0, 0), ('shuffled B=64 centre', 64, .5, 0), ('shuffled B=64 centre+pc', 64, .5, .5),
                ('shuffled B=256 centre+pc', 256, .5, .5), ('shuffled B=1024 centre+pc', 1024, .5, .5),
                ('shuffled B=64 pc only', 64, 0, .5), ('per-subject B=64 centre+pc (old assumption)', 'subj', .5, .5)]
        outs = {name: [[None] * len(subs) for _ in range(M)] for name, *_ in cfgs}
        for m in range(M):
            X = [data[s]['emb'][m] for s in subs]
            allx = np.concatenate(X); owner = np.repeat(np.arange(len(subs)), n)
            for name, B, a, b in cfgs:
                if B is None:
                    y = allx
                elif B == 'subj':
                    y = np.empty_like(allx)
                    for i in range(len(subs)):
                        o = np.flatnonzero(owner == i)[rng.permutation(n)]
                        for j in range(0, n, 64):
                            y[o[j:j + 64]] = tf(allx[o[j:j + 64]], a, b)
                else:
                    y = np.empty_like(allx); o = rng.permutation(len(allx))
                    for j in range(0, len(o), B):
                        y[o[j:j + B]] = tf(allx[o[j:j + B]], a, b)
                for i in range(len(subs)):
                    outs[name][m][i] = y[i * n:(i + 1) * n]
            # static corrections: mean output and top direction of the OTHER 4 subjects (same model), no batch use
            for i, s in enumerate(subs):
                others = np.concatenate([X[j] for j in range(len(subs)) if j != i])
                mu = others.mean(0); d = np.linalg.svd(others[rng.choice(len(others), 4000, replace=False)] - mu, full_matrices=False)[2][0]
                c = X[i] - mu
                for name, y in (('static mean (other subjects)', c + 0.5 * mu), ('static mean + static dir', c - 0.5 * np.outer(c @ d, d) + 0.5 * mu)):
                    outs.setdefault(name, [[None] * len(subs) for _ in range(M)])[m][i] = y
        for name in outs:
            for i, s in enumerate(subs):
                res.setdefault(name, {})[s] = score(dict(data[s], emb=[outs[name][m][i] for m in range(M)]), np.ones((M, n)))
    subs = sorted(data); base = np.array([res['plain'][s] for s in subs])
    print('+'.join(recipes), '(fold = 5 test subjects pooled and shuffled; deltas vs plain)')
    for name, v in res.items():
        v = np.array([v[s] for s in subs])
        print(f'  {name:44s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')


if __name__ == '__main__':
    main(sys.argv[1:])
