"""K47: how well can one trial give its subject's key, and what is it worth for the simulated re-centring of K46?
Estimators of sk_lib.key_estimators (comb / time / kron), frozen ATM, own head, normalised embeddings:
  r2      share of |m_s - K|^2 explained by the in-trial estimate
  swap    readout fed x - k + K, attention of the raw trial (K36)
  centre  e(x) - mean_j e(k + c_j) + anchor, c_j = 32 key-free training contents (9-subject averages), per query (K46)
Usage: k47_key_estimate.py <folds> [run]"""
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
    c = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs]).mean(0)
    emb = lambda v: nz(m.embed(v, fold), dim=1)
    e, mu, att = emb(x), x.mean(0, keepdim=True), atm_emb(m, x, fold)[1]
    r = dict(fold=fold, plain=top1(e, g), cal=top1(e - e.mean(0) + a, g), **{'centre oracle': top1(e - emb(mu + c).mean(0) + a, g)})
    for name, est in key_estimators(m, K).items():
        k = K + est(x - K)
        r[f'r2 {name}'] = 1 - ((k - mu).pow(2).sum((1, 2)) / (mu - K).pow(2).sum()).mean().item()
        r[f'swap {name}'] = top1(atm_emb(m, x - k + K, fold, att)[0], g)
        o = torch.stack([emb(k[i:i + 1] + c[torch.randperm(400, generator=torch.Generator().manual_seed(i))[:32]]).mean(0) for i in range(200)])
        r[f'centre {name}'] = top1(e - o + a, g)
    rows.append(r); print({k_: round(v, 2) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k47_key_estimate_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
