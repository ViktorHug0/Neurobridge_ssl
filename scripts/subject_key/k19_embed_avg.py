"""K19: remove the train / test repetition mismatch at the source. The models are trained on 4-repetition averages;
the benchmark feeds them the average of 80. Here a query is the MEAN EMBEDDING of 80 / g input averages of g
repetitions each (g = 80 is the benchmark, g = 4 is the training condition, g = 1 single trials). Same data per query,
no statistic of the test set. Each also followed by the output calibration (image anchor).
  raw: mean of the projector outputs; unit: mean of the L2-normalised embeddings.
Usage: k19_embed_avg.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, a = img_emb(m), img_emb(m, 'train').mean(0)
        for who, subs in (('held', [fold]), ('seen', [m.train_subs[(fold * 3) % 9]])):
            for s in subs:
                reps = T(np.asarray(eeg(s, 'test', average=False)))  # (200, 80, 63, 250)
                r = dict(run=RUN, fold=fold, who=who)
                for grp in (80, 16, 4, 1):
                    x = reps.view(200, 80 // grp, grp, 63, 250).mean(2).flatten(0, 1)
                    e = m.embed(x, s).view(200, 80 // grp, -1)
                    for name, q in (('raw', e.mean(1)), ('unit', nz(e, dim=2).mean(1))):
                        q = nz(q, dim=1)
                        r[f'g{grp}_{name}'], r[f'g{grp}_{name}+cal'] = top1(q, g), top1(q - q.mean(0) + a, g)
                out.append(r)
        print(RUN, fold, {k: round(v, 1) for k, v in out[-2].items() if k.endswith('raw') or k.endswith('raw+cal')}, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k19_embed_avg.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby(['run', 'who'], sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
