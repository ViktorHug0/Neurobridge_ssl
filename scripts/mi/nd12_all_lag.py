"""D14 ens5 with several members swapped for their --things_lag 5 versions at once (cached outputs, exact ranks, paired).
    nd12_all_lag.py <folds> [d6 epoch] [rmix epoch] [sqfx epoch]   swaps listed per fold only where the cache exists"""
import itertools, pandas as pd
from nc_lib import *
from nc06_names import NAMES

folds = [int(a) for a in sys.argv[1].split(',')]
EP = dict(d6=int(sys.argv[2]) if len(sys.argv) > 2 else 10, rmix=int(sys.argv[3]) if len(sys.argv) > 3 else 10, sqfx=int(sys.argv[4]) if len(sys.argv) > 4 else 40)
POS = dict(d6=0, rmix=1, sqfx=2)
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float())
gals = {'5k': gallery(), '14k': gallery(big=True)}
H = {}
for k in folds:
    F = f'{R}/logs/holdout/f{k}'
    st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    base = [Y(f'{F}/{m}.npy') for m in NAMES[k]]
    lag = {r: Y(p) for r in POS for p in [f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{EP[r]:02d}_raw.npy'] if os.path.exists(p)}
    for n in range(len(lag) + 1):
        for sw in itertools.combinations(sorted(lag), n):
            y = sum(lag[r] if r in sw else base[POS[r]] for r in POS) + base[3] + base[4]
            for gn, g in gals.items():
                h = score(y, subj, img, g, hits=True)[1]
                for rr in ('r3', 'r4'):
                    H.setdefault(('+'.join(sw) or 'none', gn, rr), {})[k] = h[rr]
rows = []
for (sw, gn, rr), v in H.items():
    if sw == 'none' or set(v) != set(H[('none', gn, rr)]) & set(v):
        continue
    a = np.concatenate([H[('none', gn, rr)][k] for k in v]); b = np.concatenate([v[k] for k in v])
    rows.append(dict(lag_members=sw, folds=','.join(map(str, v)), cell=f'{gn} {rr}', base=100 * a.mean(), lag=100 * b.mean(), gain='%+.1f +- %.1f %%' % paired(b, a)))
print(pd.DataFrame(rows).sort_values(['cell', 'lag_members']).round(3).to_string(index=False))
