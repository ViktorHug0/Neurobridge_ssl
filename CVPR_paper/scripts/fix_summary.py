"""Summarise a regimes_fix* rows CSV set: per-cell means of every method, the all-cell mean, and one config selected
CROSS-FOLD (best all-cell mean on the other subjects) per held-out subject.
usage: fix_summary.py 'glob/of/rows*.csv'
"""
import glob, sys

import pandas as pd

df = pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1])])
df = df[~df.method.str.startswith('stat_')]
df['cell'] = [f'D{l}' if r == 'R4' else f'{law[:3]}{l}' for r, law, l in zip(df.regime, df.law, df.level)]
t = df.groupby(['subject', 'method', 'cell']).top1.mean().unstack()  # mean over draws
t = t[sorted(t.columns, key=lambda c: (c[0] != 'D', c[:3], int(c[3:] if c[0] != 'D' else c[1:])))]
t['MEAN'] = t.mean(1)
print(f'subjects: {sorted(t.index.get_level_values(0).unique())}')
m = t.groupby(level='method').mean().sort_values('MEAN', ascending=False)
print(m.round(1).to_string())
subj = t['MEAN'].unstack()
cv = []
for s in subj.index:
    best = subj.drop(index=s).mean().drop(['plain', 'score']).idxmax()
    cv.append(t.loc[(s, best)])
    print(f'sub {s}: picked {best}')
print('cross-fold selected config:')
print(pd.DataFrame(cv).mean().round(1).to_string())
