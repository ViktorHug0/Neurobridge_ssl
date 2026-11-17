"""D06 finer dataset lag: Alljoined-only REVE-d6 (fold 3, ep10) on five other THINGS people (sub-22, 24, 26, 28, 30):
global lag 5 / 5.5 / 6 vs the per-person lag of D04 (rounded, and linearly interpolated)."""
import pandas as pd
from nc_lib import *
from nd03_cross_dataset_lib import shift

f = load('reve_d6_ho5k_mid_optx_aj', 3, 'checkpoint_ep10.pth')
gal = gallery()
people = [22, 24, 26, 28, 30]
X, subj, img = cell(people)
sid = torch.as_tensor(subj).to(DEV)
plag = dict(np.load(f'{OUT}/d04_person_lag.npy').tolist())
frac = lambda x, v: (1 - (v - int(np.floor(v)))) * shift(x, int(np.floor(v))) + (v - int(np.floor(v))) * shift(x, int(np.floor(v)) + 1)


def per_person(x, a, b, rnd):
    out = x.clone()
    for s in people:
        m = sid[a:b] == s
        if m.any():
            out[m] = shift(x[m], int(round(plag[s]))) if rnd else frac(x[m], plag[s])
    return out


conds = {'lag 5': lambda x, a, b: shift(x, 5), 'lag 5.5': lambda x, a, b: frac(x, 5.5), 'lag 6': lambda x, a, b: shift(x, 6),
         'per-person lag, rounded': lambda x, a, b: per_person(x, a, b, True), 'per-person lag, interpolated': lambda x, a, b: per_person(x, a, b, False)}
rows = []
for cn, T in conds.items():
    y = f(X, lambda a, b: T(torch.as_tensor(X[a:b]).to(DEV), a, b) - torch.as_tensor(X[a:b]).to(DEV))
    rows.append(dict(cond=cn, **score(y, subj, img, gal))); print(rows[-1], flush=True)
print({s: round(plag[s], 2) for s in people})
print(pd.DataFrame(rows).round(3).to_string(index=False))
