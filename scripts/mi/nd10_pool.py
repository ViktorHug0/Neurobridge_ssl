"""D12 lag rmix (ep10), folds pooled (logs/holdout/f<k> streams, cached outputs): hits of both folds concatenated,
paired bootstrap over (person, image) groups. rmix alone, ens5 with rmix swapped, pair rmix + .5 SQF-X with rmix swapped;
with SQFLAG=1 also SQF-X swapped for its lag version where it exists."""
import pandas as pd
from nc_lib import *
from nc06_names import NAMES

folds = [int(a) for a in sys.argv[1].split(',')]
both = os.environ.get('SQFLAG')
gals = {'5k': gallery(), '14k': gallery(big=True)}
H = {}
for k in folds:
    F = f'{R}/logs/holdout/f{k}'
    st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    Y = lambda p: scaled(torch.as_tensor(np.load(p)).float())
    D6, RM, SX, SC, SP = (Y(f'{F}/{m}.npy') for m in NAMES[k])
    RL = Y(f'{OUT}/hold_f{k}_mi_rmix_ho5k_lag5_ep10_raw.npy')
    SL = Y(f'{OUT}/hold_f{k}_mi_sqfx_ho5k_lag5_ep40_raw.npy') if both else SX
    rows = {'rmix alone': (RM, RL), 'ens5': (D6 + RM + SX + SC + SP, D6 + RL + SL + SC + SP), 'pair rmix + .5 SQF-X': (RM + .5 * SX, RL + .5 * SL),
            'REVE trio (d6 + rmix + scr)': (D6 + RM + SC, D6 + RL + SC)}
    for n, (a, b) in rows.items():
        for gn, g in gals.items():
            ha, hb = score(a, subj, img, g, hits=True)[1], score(b, subj, img, g, hits=True)[1]
            for rr in ('r3', 'r4'):
                H.setdefault((n, gn, rr), []).append((ha[rr], hb[rr]))
out = []
for (n, gn, rr), v in H.items():
    a, b = np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])
    g, se = paired(b, a)
    out.append(dict(row=n, cell=f'{gn} {rr}', base=100 * a.mean(), lag=100 * b.mean(), gain='%+.1f +- %.1f %%' % (g, se),
                    per_fold=' '.join('%+.0f' % (100 * (y.mean() / x.mean() - 1)) for x, y in v)))
df = pd.DataFrame(out)
print(df.round(3).to_string(index=False))
for n in df.row.unique():
    a = np.concatenate([np.concatenate([x[0] for x in H[(n, gn, rr)]]) for gn in gals for rr in ('r3', 'r4')])
    b = np.concatenate([np.concatenate([x[1] for x in H[(n, gn, rr)]]) for gn in gals for rr in ('r3', 'r4')])
    print('%-28s all four cells pooled (hits): %+.1f %%' % (n, 100 * (b.mean() / a.mean() - 1)))
