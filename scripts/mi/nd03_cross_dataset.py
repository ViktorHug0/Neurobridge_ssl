"""D03 functional test of the dataset mismatch. An Alljoined-ONLY model (fold 3, *_ho5k_mid_optx_aj, ep10) decodes
THINGS-EEG2 people (never seen: another dataset) on the 5,000 never-trained images, 5k gallery, top-5 % r=1/3/4, after
transforming their trials: lag (shifted later by n samples), gain, dataset-key swap (x - K_TH + K_AJ, after the lag).
Reference rows: the same model on its own held-out Alljoined people.
    nd03_cross_dataset.py <recipe> [reverse]   reverse: a THINGS+Alljoined model is not needed; instead Alljoined
    held-out people are shifted EARLIER through the Alljoined-only model as a sanity check of its lag sensitivity."""
import pandas as pd
from nc_lib import *

k, recipe = 3, sys.argv[1] if len(sys.argv) > 1 else 'reve_d6_ho5k_mid_optx_aj'
f = load(recipe, k, 'checkpoint_ep10.pth')
gal = gallery()
AJtrain = [s for s in range(1, 21) if s not in FOLDS[k]]
K_AJ = torch.stack([key(s) for s in AJtrain]).mean(0).to(DEV)
K_TH = torch.stack([key(s) for s in range(21, 31)]).mean(0).to(DEV)


def shift(x, n):  # x(t) -> x(t - n): later by n samples (n < 0: earlier); edges replicate
    if n == 0:
        return x
    return torch.cat([x[..., :1].expand(*x.shape[:-1], n), x[..., :-n]], -1) if n > 0 else torch.cat([x[..., -n:], x[..., -1:].expand(*x.shape[:-1], -n)], -1)


rows = []
def run(name, X, subj, img, T):
    r = dict(cond=name, **score(f(X, None if T is None else (lambda a, b: T(torch.as_tensor(X[a:b]).to(DEV)) - torch.as_tensor(X[a:b]).to(DEV))), subj, img, gal))
    rows.append(r); print(r, flush=True)


X, subj, img = cell(FOLDS[k])
run('Alljoined held-out people (reference)', X, subj, img, None)
for n in (-4, -2, 2, 4):
    run(f'Alljoined held-out, lag {n:+d}', X, subj, img, lambda x, n=n: shift(x, n))
del X
X, subj, img = cell([21, 23, 25, 27, 29])
run('THINGS people raw', X, subj, img, None)
for n in (2, 3, 4, 5, 6, 7, 8, 10):
    run(f'THINGS lag {n:+d}', X, subj, img, lambda x, n=n: shift(x, n))
best = max((r for r in rows if r['cond'].startswith('THINGS lag')), key=lambda r: r['r3'])
n = int(best['cond'].split()[-1])
for g in (0.33, 0.5, 2.0):
    run(f'THINGS lag {n:+d}, gain x{g}', X, subj, img, lambda x: g * shift(x, n))
run(f'THINGS lag {n:+d}, key -> K_AJ', X, subj, img, lambda x: shift(x - K_TH, n) + K_AJ)
run(f'THINGS lag {n:+d}, key x0.33', X, subj, img, lambda x: shift(x - 0.67 * K_TH, n))
run('THINGS raw, key -> K_AJ', X, subj, img, lambda x: x - K_TH + K_AJ)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nd03_{recipe}.csv', index=False); print(df.round(3).to_string(index=False))
