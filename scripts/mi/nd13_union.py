"""D16 union of lag and non-lag members (cached outputs, fold 1 and fold 3 where available): ens5 with lag swaps vs
ens5 + the lag versions as extra members (weights: every member 1, or each lag/non-lag pair sharing weight 1)."""
import pandas as pd
from nc_lib import *
from nc06_names import NAMES
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float())
gal = gallery(); H = {}
for k in [int(a) for a in sys.argv[1].split(',')]:
    F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    base = [Y(f'{F}/{m}.npy') for m in NAMES[k]]
    lag = {i: Y(p) for i, r, e in ((0, 'd6', 10), (1, 'rmix', 10), (2, 'sqfx', 40)) for p in [f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy'] if os.path.exists(p)}
    sets = {'ens5': sum(base), 'swapped': sum(lag.get(i, b) for i, b in enumerate(base)), 'union, all weight 1': sum(base) + sum(lag.values()),
            'union, pairs share weight 1': sum(.5 * (b + lag[i]) if i in lag else b for i, b in enumerate(base))}
    for n, y in sets.items():
        h = score(y, subj, img, gal, hits=True)[1]
        for rr in ('r3', 'r4'):
            H.setdefault((n, rr), []).append(h[rr])
    print(k, sorted(lag), flush=True)
for (n, rr), v in H.items():
    a = np.concatenate(v); b = np.concatenate(H[('ens5', rr)])
    print('%-30s %s  %.3f  %+.1f +- %.1f %%' % (n, rr, 100 * a.mean(), *paired(a, b)))
