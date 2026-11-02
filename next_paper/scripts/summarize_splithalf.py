"""Split-half selection of the trust level r: full top-1 at argmax_r agree, vs fixed r and the per-subject oracle."""
import glob, os, sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C  # noqa: E402

d = pd.concat([pd.read_csv(f).assign(run=os.path.basename(f)) for f in sorted(glob.glob(f'{C}/splithalf_atm128_vc_s*.csv'))])
t = d.pivot_table(index=['run', 'subject', 'D', 'r'], columns='metric', values='value').reset_index()
rows = []
for (run, sid, D), g in t.groupby(['run', 'subject', 'D']):
    g = g.reset_index(drop=True)
    i = g.agree.idxmax()
    rows.append({'D': D, 'oracle': g.top1.max(), 'gala': g.top1[0], 'cons': g.top1.iloc[-1], 'sel_agree': g.top1[i],
                 'r_sel': g.r[i], 'r_oracle': g.r[g.top1.idxmax()], 'rho_agree_top1': spearmanr(g.agree, g.top1)[0],
                 'sqrt_agree': 10 * np.sqrt(g.agree[i]), 'half_acc': g.half_acc[i]})
s = pd.DataFrame(rows)
print(s.groupby('D').mean().T.round(2))
print('runs:', s.groupby('D').size().to_dict())
