"""K46: a pure inductive re-centring? Re-centring (K10, 48.6) subtracts the subject's mean output embedding, which needs
its other trials. K39: the embedding of the key alone is not that mean (the model is not linear). Here the mean is
SIMULATED: the subject's key carried by generic content from the training subjects,
    o = mean_j e(k + c_j),   c_j = key-free training trials,   query scored as  e(x) - o + anchor
  c4    c_j = one training subject's 4-repetition trial minus its key
  c36   c_j = the same image averaged over the 9 training subjects (36 repetitions: closer to the noise level of a test query)
  key m k = the subject's mean test ERP (test data required: does the simulation give the right centre at all?)
  key x k = the in-trial key of each query, K + comb(x - K) (pure inductive), J = 32 contents per query
Frozen ATM, normalised embeddings, own head. Reference: cal = the true mean (K10), comb = K36.
Usage: k46_synthetic_centre.py <folds> [run]"""
import pandas as pd
from sk_lib import *

FOLDS = [int(f) for f in sys.argv[1].split(',')]
RUN = sys.argv[2] if len(sys.argv) > 2 else 'atm128_vc/seed3300'
IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))
rows = []
for fold in FOLDS:
    m = load(RUN, fold)
    g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
    a = img_emb(m, 'train').mean(0)
    C = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs])  # (9, 400, 63, 250)
    c4, c36 = C[torch.arange(400) % 9, torch.arange(400)], C.mean(0)
    per = (x - K).view(200, 63, 5, 50).mean(2)
    k = K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)
    emb = lambda v: nz(m.embed(v, fold), dim=1)
    e, mu = emb(x), x.mean(0, keepdim=True)
    r = dict(fold=fold, plain=top1(e, g), cal=top1(e - e.mean(0) + a, g))
    for name, c in (('c4', c4), ('c36', c36)):
        o = emb(mu + c).mean(0)
        r[f'{name} key m'] = top1(e - o + a, g)
        r[f'{name} key m: cos with the true centre'] = torch.cosine_similarity(o - a, e.mean(0) - a, dim=0).item()
        o = torch.stack([emb(k[i:i + 1] + c[torch.randperm(400, generator=torch.Generator().manual_seed(i))[:32]]).mean(0) for i in range(200)])
        r[f'{name} key x'] = top1(e - o + a, g)
    rows.append(r); print({k_: round(v, 2) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k46_synthetic_centre_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
