"""Summarize a decoy selection run: within-cell rank correlation of the decoy-hit rate with top-1, and the top-1 of
decoy-selected vs fixed candidates. usage: summarize_decoy.py <tag, e.g. decoyfit100>"""
import glob, sys
import pandas as pd
from scipy.stats import spearmanr
tag = sys.argv[1]
d = pd.concat([pd.read_csv(f).assign(seed=f[-8:-4]) for f in glob.glob(f'results/things_eeg/cvpr/next_paper/open_gallery/{tag}_atm128_vc_s*.csv')])
r = d.groupby(['seed', 'subject', 'D'])[['decoy_hit', 'top1']].apply(lambda x: spearmanr(x.decoy_hit, x.top1)[0]).groupby('D').mean()
print(tag, 'rho(decoy_hit, top1):', r.round(2).to_dict())
o = d.assign(si=d.start.map({'saw': 0, 'half': 1, 'mm': 2})).sort_values(['decoy_hit', 'lam', 'si'])
m = lambda x: x.groupby(['seed', 'D']).top1.mean().groupby('D').mean()
out = {'decoy-selected': m(o.groupby(['seed', 'subject', 'D']).head(1)),
       'oracle-selected': m(d.sort_values('top1', ascending=False).groupby(['seed', 'subject', 'D']).head(1))}
for st, l in [('saw', 0.0), ('half', 1.0), ('mm', 1.0)]:
    out[f'{st},{l}'] = m(d[(d.start == st) & (d.lam == l)])
print(pd.DataFrame(out).T.round(1))
