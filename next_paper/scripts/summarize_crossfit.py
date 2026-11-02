"""Top-1 of F(r) with r chosen per subject by each label-free criterion (argmax), vs fixed r and the per-subject oracle.
usage: summarize_crossfit.py [glob]
"""
import glob, os, sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C  # noqa: E402

files = sorted(glob.glob(f'{C}/' + (sys.argv[1] if len(sys.argv) > 1 else 'crossfit_atm128_vc_s*.csv')))
d = pd.concat([pd.read_csv(f).assign(run=os.path.basename(f)) for f in files])
t = d.pivot_table(index=['run', 'subject', 'D', 'r'], columns='metric', values='value').reset_index()
rows = []
for (run, sid, D), g in t.groupby(['run', 'subject', 'D']):
    row = {'D': D, 'oracle': g.top1.max()}
    for r in sorted(t.r.unique()):
        row[f'r={r}'] = g.loc[g.r == r, 'top1'].item()
    for m in [c for c in t.columns if c.startswith(('cv_', 'in_'))]:
        row[m] = g.loc[g[m].idxmax(), 'top1']
        row[f'rho_{m}'] = spearmanr(g[m], g.top1)[0]
        row[f'rsel_{m}'] = g.loc[g[m].idxmax(), 'r']
    rows.append(row)
s = pd.DataFrame(rows).groupby('D').mean()
pd.set_option('display.width', 250)
print(s[[c for c in s.columns if not c.startswith(('rho_', 'rsel_'))]].T.round(1))
print(s[[c for c in s.columns if c.startswith('rho_')]].T.round(2))
print(s[[c for c in s.columns if c.startswith('rsel_')]].T.round(2))
