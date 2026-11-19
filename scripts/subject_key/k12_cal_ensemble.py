"""K12: score ensembles of CALIBRATED members (K10, image anchor): uniform mean of the calibrated cosine matrices.
Same pools as K01. Cache: results/things_eeg/subject_key/scores/<run>_cal.pt"""
import itertools
import pandas as pd
from sk_lib import *

S = {}
for run in sum(POOLS.values(), []) + ['eegnet_vc/seed3300']:
    p = f'{SK}/scores/{run.replace("/", "_")}_cal.pt'
    if not os.path.exists(p):
        torch.save(torch.stack([calibrated(load(run, f), f) for f in range(1, 11)]).half(), p)
    S[run] = torch.load(p).float()
    print(f'{run:28s} cal {acc(S[run]).mean():.2f}', flush=True)
rows = []
for pool, runs in POOLS.items():
    for k in range(1, 6):
        for combo in itertools.combinations(runs, k):
            a = acc(sum(S[c] for c in combo))
            rows.append(dict(pool=pool, k=k, members=' + '.join(c.replace('/seed', ':') for c in combo), cal=a.mean().item(), folds=' '.join(f'{v:.0f}' for v in a)))
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k12_cal_ensemble.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 130)
for pool in POOLS:
    print(df[df.pool == pool].sort_values('cal', ascending=False).groupby('k').head(4).sort_values(['k', 'cal'], ascending=[True, False]).round(2).to_string(index=False))
