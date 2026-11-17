"""Pooled multi-fold read of a stack saved by `SAVE=<tag> nu04_stack.py <fold>`: the six with every swap against the reference six,
relative change in % with a paired bootstrap over queries (folds pooled), per-fold top-5, 5k and 14k galleries, r=3.
The forecast of a package is read here, on all the folds given.
    nu06_pool.py <tag> <folds, e.g. 1,2,3,4>"""
import sys, numpy as np
OUT = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/mi/nc'; tag, folds = sys.argv[1], [int(a) for a in sys.argv[2].split(',')]
d = [np.load(f'{OUT}/stack_{tag}_f{k}.npz') for k in folds]
for name, b, n, G in (('5k ', 'base', 'new', 4999.), ('14k', 'base14', 'new14', 14365.)):
    B, N = np.concatenate([x[b] for x in d]), np.concatenate([x[n] for x in d]); bi = np.random.default_rng(0).integers(0, len(B), (1000, len(B))); o = []
    for lab, fn in (('top-1', lambda v: v < 1), ('top-2', lambda v: v < 2), ('top-3', lambda v: v < 3), ('top-5', lambda v: v < 5), ('top-10', lambda v: v < 10), ('top-20', lambda v: v < 20), ('top-100', lambda v: v < 100), ('ranking error', lambda v: v / G)):
        a, c = fn(N).astype(float), fn(B).astype(float); o.append('%s %+.1f +- %.1f' % (lab, 100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
    print(name, ' | '.join(o), '| top-5 per fold', ' / '.join('%+.1f' % (100 * ((x[n] < 5).mean() / (x[b] < 5).mean() - 1)) for x in d), '| reference top-5 %.3f %%' % (100 * (B < 5).mean()))
