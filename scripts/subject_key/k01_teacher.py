"""K01: which inductive ensemble can serve as the teacher? Score matrices of every candidate member on the held-out
subject (own key / substituted), cached in results/things_eeg/subject_key/scores/<arm>_<seed>.pt, then every subset of
up to 4 members (uniform mean of cosine matrices). Two pools: validation-selected (*_vc) and test-selected runs."""
import itertools
import pandas as pd
from sk_lib import *

os.makedirs(f'{SK}/scores', exist_ok=True)
S = {}
for run in sum(POOLS.values(), []):
    p = f'{SK}/scores/{run.replace("/", "_")}.pt'
    if not os.path.exists(p):
        own, sub = zip(*[member_scores(load(run, f), f) for f in range(1, 11)])
        torch.save({'own': torch.stack(own).half(), 'sub': torch.stack(sub).half()}, p)
    S[run] = {k: v.float() for k, v in torch.load(p).items()}
    print(f'{run:28s} own {acc(S[run]["own"]).mean():.2f}  sub {acc(S[run]["sub"]).mean():.2f}', flush=True)
rows = []
for pool, runs in POOLS.items():
    for k in range(1, 5):
        for combo in itertools.combinations(runs, k):
            r = dict(pool=pool, k=k, members=' + '.join(c.replace('/seed', ':') for c in combo))
            for mode in ('own', 'sub'):
                a = acc(sum(S[c][mode] for c in combo))
                r[mode] = a.mean().item()
                if mode == 'sub':
                    r['folds'] = ' '.join(f'{v:.0f}' for v in a)
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k01_teacher.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
for pool in POOLS:
    print(df[df.pool == pool].sort_values('sub', ascending=False).groupby('k').head(6).sort_values(['k', 'sub'], ascending=[True, False]).round(2).to_string(index=False))
