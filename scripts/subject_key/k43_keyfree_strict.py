"""K43: models TRAINED without the key (train.py --subject_key demean: every trial minus its subject's mean ERP), scored
on the held-out subject with a key that needs none of its other trials. Input x - k:
  oracle   k = the subject's mean ERP over its test trials (as trained; test data required)
  K        k = the mean training key (pure inductive, no paradigm knowledge): the subject's key deviation stays in the input
  comb     k = K + the 200 ms-periodic part of x - K (K36; pure inductive)
Usage: k43_keyfree_strict.py <absolute run dir> [...]"""
import pandas as pd
from sk_lib import *

rows = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        if not glob.glob(f'{RUN}/*-sub-{fold:02d}/checkpoint_test_best.pth'):
            continue
        m = load(RUN, fold)
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        per = (x - K).view(200, 63, 5, 50).mean(2)
        ks = {'oracle': x.mean(0, keepdim=True), 'K': K, 'comb': K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)}
        rows.append(dict(run=os.path.basename(RUN), fold=fold, **{n: top1(m.embed(x - k, fold), g) for n, k in ks.items()}))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k43_keyfree_strict.csv', index=False)
print(df.groupby('run', sort=False).agg(['mean', 'count']).drop(columns='fold').round(2).to_string())
