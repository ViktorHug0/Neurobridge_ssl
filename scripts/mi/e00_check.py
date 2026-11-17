"""Check: the staged forward reproduces each fold's `best top1 acc`, and measure the seen-subject ceiling
(training subjects' accuracy on the 200 test images, which the model never saw) from the same checkpoint."""
import pandas as pd
from mi_lib import *

rows = []
for run in RUNS:
    for sub in range(1, 11):
        m = M(run, sub)
        g = img_emb(m)
        acc = {s: top1(forward_from(m, 'input', T(eeg(s))), g) for s in range(1, 11)}
        ref = pd.read_csv(f'{run_dir(run, sub)}/result.csv')['best top1 acc'][0]
        seen = np.mean([acc[s] for s in m.train_subs])
        rows.append(dict(run=run, fold=sub, ref=ref, heldout=acc[sub], seen_mean=seen, seen_min=min(acc[s] for s in m.train_subs)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
assert (df.ref - df.heldout).abs().max() < 1e-3, 'staged forward does not reproduce result.csv'
os.makedirs(OUT, exist_ok=True)
df.to_csv(f'{OUT}/e00_check.csv', index=False)
print(df.groupby('run')[['ref', 'heldout', 'seen_mean']].mean())
