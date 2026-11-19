"""K30: a strictly inductive shift. The re-centring needs the new subject's mean embedding. Here the shift is FIXED,
estimated on training data only: d = (mean image embedding of the training images) - (mean embedding of the
TRAINING subjects' training trials), added to every query of the new subject. It can only remove what is common to
all subjects (the gap between the EEG and image clouds), not the subject's own offset.
  g80: the benchmark query (embedding of the 80-repetition average); g4: mean embedding of twenty 4-repetition averages
  (the training condition, where the fixed shift is exactly the training-time gap).
Columns: plain | +fixed shift | +own re-centring (reference, needs the subject's unlabelled trials).
Usage: k30_fixed_shift.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 1500, replace=False))
out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, a = img_emb(m), img_emb(m, 'train').mean(0)
        d = a - torch.cat([nz(m.embed(T(eeg(s, 'train')[IM]), s), dim=1) for s in m.train_subs]).mean(0)
        reps = T(np.asarray(eeg(fold, 'test', average=False)))
        r = dict(run=RUN, fold=fold)
        for grp in (80, 4):
            e = m.embed(reps.view(200, 80 // grp, grp, 63, 250).mean(2).flatten(0, 1), fold).view(200, 80 // grp, -1)
            q = nz(nz(e, dim=2).mean(1), dim=1)
            r[f'g{grp}'], r[f'g{grp}+fixed'], r[f'g{grp}+own'] = top1(q, g), top1(q + d, g), top1(q - q.mean(0) + a, g)
            for w in (0.5, 2.0):
                r[f'g{grp}+fixed x{w}'] = top1(q + w * d, g)
        out.append(r)
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k30_fixed_shift.csv', index=False)
pd.set_option('display.width', 220)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
