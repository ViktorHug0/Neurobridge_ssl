"""K38: the in-trial key (K36) on the other encoders. Frozen model, held-out subject, pure inductive setting.
Input x - lam * (k - K): lam = 1 replaces the key by the mean training key K, lam = 0.5 moves it halfway (E10's midpoint).
k = the in-trial key (comb: K + the 200 ms-periodic part of x - K, without its temporal mean) or the subject's mean ERP
over its test trials (oracle: test data required).
Usage: k38_self_key_encoders.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

rows = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        per = (x - K).view(200, 63, 5, 50).mean(2)
        d = {'comb': (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5), 'oracle': x.mean(0, keepdim=True) - K}
        rows.append(dict(run=RUN, fold=fold, plain=top1(m.embed(x, fold), g),
                         **{f'{n} {lam}': top1(m.embed(x - lam * v, fold), g) for n, v in d.items() for lam in (0.5, 1.0)}))
        print(RUN, fold, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k38_self_key_encoders.csv', index=False)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
