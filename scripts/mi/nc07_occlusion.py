"""N07 which part of the trial carries the image content, for seen vs new people? Content ablation: inside one time
window or electrode group the trial is replaced by the person's key (mean ERP), so the person cue stays intact and only
the trial-specific signal is removed. top-5 % r=3, 5k gallery, never-trained images. Epoch = -0.2..0.8 s at 120 Hz."""
import pandas as pd
from nc_lib import *

k, member = int(os.environ.get('K', 1)), sys.argv[1] if len(sys.argv) > 1 else 'd6'
NEW, SEENP = FOLDS[k], FOLDS[k % 4 + 1]
CH = json.load(open(f'{DATA}/info.json'))['ch_names']
grp = lambda pref: [i for i, c in enumerate(CH) if c.startswith(pref)]
T = lambda a, b: slice(24 + round(a * 120), 24 + round(b * 120))
ALL = slice(None)
conds = {'none': None, 'prestim': (ALL, slice(0, 24)), '0-100 ms': (ALL, T(0, .1)), '100-200 ms': (ALL, T(.1, .2)), '200-300 ms': (ALL, T(.2, .3)),
         '300-500 ms': (ALL, T(.3, .5)), '500-800 ms': (ALL, T(.5, .8)), 'after 300 ms': (ALL, T(.3, .8)), 'after 500 ms': (ALL, T(.5, .8)),
         'O + PO': (grp(('O', 'PO')), ALL), 'P': ([i for i, c in enumerate(CH) if c[0] == 'P' and not c.startswith('PO')], ALL), 'CP': (grp('CP'), ALL),
         'frontal + central': (grp(('F', 'AF', 'C')) if False else [i for i, c in enumerate(CH) if c[0] in 'FA' or c in ('Cz',)], ALL)}
keys = {s: key(s).to(DEV) for s in range(1, 21)}
own = torch.stack([keys.get(s, keys[1]) for s in range(21)])
f = load(member, k)
gal = gallery()
rows = []
for pn, people in (('new', NEW), ('seen', SEENP)):
    X, subj, img = cell(people)
    sid = torch.as_tensor(subj).to(DEV)
    for cn, c in conds.items():
        def delta(a, b):  # x + mask * (key - x)
            x = torch.as_tensor(X[a:b]).to(DEV); d = torch.zeros_like(x)
            d[:, c[0], c[1]] = (own[sid[a:b]] - x)[:, c[0], c[1]]
            return d
        rows.append(dict(person=pn, ablate=cn, r3=score(f(X, None if c is None else delta), subj, img, gal)['r3']))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows).pivot(index='ablate', columns='person', values='r3').loc[list(conds)]
df['new rel'] = df.new / df.new['none']; df['seen rel'] = df.seen / df.seen['none']
df.to_csv(f'{OUT}/nc07_occlusion_{member}_f{k}.csv'); print(df.round(3).to_string())
