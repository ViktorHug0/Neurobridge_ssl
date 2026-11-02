"""E8.2 complementarity: is the SAGE gain larger for weaker zero-shot (R0) decoders?
Per-subject (plain, SAGE) from every r1_eval table (inter: final P-main arms + ValCon arms; intra: atm128_oc), and per-q
means from the repetitions sweeps. Reports Spearman rho of gain vs R0 (a) within each model over subjects, (b) pooled
over subjects after removing each model's mean (within-model), (c) across models (model means), (d) across q.
usage: complementarity.py
"""
import glob, os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = 'results/things_eeg/cvpr'
tables = {os.path.basename(os.path.dirname(f)): f for f in glob.glob(f'{R}/final/*/r1_competitors.csv')}
tables.update({os.path.basename(f)[:-4]: f for f in glob.glob(f'{R}/r1_competitors/*.csv')})
tables['intra_atm128_oc'] = f'{R}/intra_analyses/atm128_oc_s2025/r1_competitors.csv'
rows = []
for name, f in sorted(tables.items()):
    t = pd.read_csv(f).pivot(index='subject', columns='method', values='top1')
    rows += [{'model': name, 'subject': s, 'r0': r.plain, 'gain': r.sage - r.plain} for s, r in t.iterrows()]
d = pd.DataFrame(rows)
print('(a) within model, over subjects')
for m, g in d.groupby('model'):
    rho, p = spearmanr(g.r0, g.gain)
    print(f'  {m:22s} R0 {g.r0.mean():5.1f}  gain {g.gain.mean():5.1f}  rho {rho:+.2f}  p {p:.3f}')
inter = d[~d.model.str.startswith('intra')]
c = inter.assign(r0=inter.r0 - inter.groupby('model').r0.transform('mean'), gain=inter.gain - inter.groupby('model').gain.transform('mean'))
rho, p = spearmanr(c.r0, c.gain)
print(f'(b) pooled within-model (inter, n={len(c)}): rho {rho:+.2f} p {p:.3g}')
mm = inter.groupby('model')[['r0', 'gain']].mean()
rho, p = spearmanr(mm.r0, mm.gain)
print(f'(c) across inter models (n={len(mm)}): rho {rho:+.2f} p {p:.3f}')
print('(d) across q (repetitions sweep, model means)')
for f in sorted(glob.glob(f'{R}/final/*/reps.csv')):
    r = pd.read_csv(f).groupby(['reps', 'method']).top1.mean().unstack()
    print(f'  {f.split("/")[-2]:14s}', ' '.join(f'q{q}: R0 {r.loc[q, "plain"]:.1f} gain {r.loc[q, "full"] - r.loc[q, "plain"]:+.1f}' for q in r.index))
