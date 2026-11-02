"""Summarise stream_grid.py rows: per method, prequential accuracy (mean of `<method>@new` over menus, reps, orders and
checkpoints), the first checkpoint's buffer score and the end-of-stream buffer score; plus each arm minus `sage_a1`
(subjects improved, Wilcoxon p on prequential).
usage: stream_summary.py 'glob/of/stream_rows.csv'
"""
import glob, sys

import pandas as pd
from scipy.stats import wilcoxon

d = pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1])])
key = ['subject', 'menu', 'reps', 'seed']
d['first'] = d.t == d.groupby(key).t.transform('min')
d['last'] = d.t == d.groupby(key).t.transform('max')
new = d[d.method.str.endswith('@new')].assign(method=lambda x: x.method.str[:-4])
buf = d[~d.method.str.endswith('@new')]
per = pd.DataFrame({'preq': new.groupby(['subject', 'method']).top1.mean(),
                    'first': buf[buf['first']].groupby(['subject', 'method']).top1.mean(),
                    'end': buf[buf['last']].groupby(['subject', 'method']).top1.mean()})
print(f'subjects {d.subject.nunique()}')
print(per.groupby(level='method').mean().round(1).to_string())
pq = per.preq.unstack()
for m in pq.columns.drop(['sage_a1', 'plain'], errors='ignore'):
    diff = pq[m] - pq['sage_a1']
    print(f'{m} - sage_a1 prequential {diff.mean():+.2f}  {int((diff > 0).sum())}/{len(diff)}  p={wilcoxon(diff).pvalue:.3g}')
