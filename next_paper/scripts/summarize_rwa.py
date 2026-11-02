"""RWA vs ACEF over encoders x D (rwa.py outputs): per-D means, and per-cell differences (cell = encoder x D)."""
import glob, os, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C  # noqa: E402

d = pd.concat([pd.read_csv(f).assign(arm=os.path.basename(f)[4:-10]) for f in sorted(glob.glob(f'{C}/rwa_*_s3300.csv'))])
m = d[~d.method.str.startswith('agree_')].groupby(['arm', 'D', 'method']).top1.mean().unstack()
pd.set_option('display.width', 250)
print(m.round(1).to_string())
print('\nmean over encoders by D')
print(m.groupby('D').mean().round(1).T.to_string())
for k in ('rwa1', 'rwa2', 'rwa4', 'sel'):
    diff = m[k] - m['acef']
    best = m[['F0', 'F.5', 'F1', 'S']].max(1)
    print(f'{k} - acef: mean {diff.mean():+.2f}  worst {diff.min():+.2f}  wins {(diff > 0).sum()}/{len(diff)}   '
          f'vs best member: mean {(m[k] - best).mean():+.2f} worst {(m[k] - best).min():+.2f}')
