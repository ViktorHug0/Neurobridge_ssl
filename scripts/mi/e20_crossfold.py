"""Cross-fold selection of the self-training hyper-parameters (as the paper does for GALA): for each held-out subject,
take the (learning rate, number of rounds) with the best mean head accuracy over the OTHER nine subjects.
usage: e20_crossfold.py <tag prefix> <lr> <lr> ...   reads results/things_eeg/mi/e20_att_tta_<prefix><lr>.csv"""
import sys
import numpy as np, pandas as pd
from mi_lib import OUT

pre, lrs = sys.argv[1], sys.argv[2:]
d = pd.concat([pd.read_csv(f'{OUT}/e20_att_tta_{pre}{lr}.csv').assign(lr=lr) for lr in lrs])
t = d.pivot_table(index='fold', columns=['lr', 'round'], values='tta')
base = d[d['round'] == 0].groupby('fold').tta.first()
sel, picks = [], []
for f in t.index:
    best = t.drop(index=f).mean().idxmax()
    sel.append(t.loc[f, best]); picks.append(best)
print('per lr, last round :', {lr: round(t[lr].iloc[:, -1].mean(), 2) for lr in lrs})
print('before %.2f | cross-fold selected %.2f | oracle single config %.2f | picks %s' % (base.mean(), np.mean(sel), t.mean().max(), sorted(set(picks))))
print('per subject: before', base.round(1).tolist(), '\n             after ', np.round(sel, 1).tolist(), ' up:', int((np.array(sel) > base.values).sum()), '/ 10')
