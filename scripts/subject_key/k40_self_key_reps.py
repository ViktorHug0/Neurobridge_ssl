"""K40: the in-trial key (K36) when queries have few repetitions. As K37 (frozen ATM trunk, three distilled heads, no
re-centring, pure inductive), query = mean of the first r repetitions, r = 1, 4, 16, 80; the key is estimated in that query.
Usage: k40_self_key_reps.py [trunk]"""
import pandas as pd
from sk_lib import *

TRUNK = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
TN = TRUNK.replace('/', '_')
rows = []
for fold in range(1, 11):
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    X = torch.load(f'{SK}/cache/trunk_{TN}_f{fold}.pt')['tr'].double(); mu = X.mean(0); X -= mu
    XtX = X.T @ X
    A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
    W = [(A @ (X.T @ (t['eeg'][k].double() - t['eeg'][k].double().mean(0))), t['eeg'][k].double().mean(0)) for k in (0, 2, 3)]
    m = load(TRUNK, fold)
    pre = []
    m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.append(i[0].detach().double().cpu()))
    g, K, reps = img_emb(m), mean_key(m), np.asarray(eeg(fold, 'test', average=False))
    r = dict(fold=fold)
    for n in (1, 4, 16, 80):
        x = T(reps[:, :n].mean(1))
        per = (x - K).view(200, 63, 5, 50).mean(2)
        att = atm_emb(m, x, fold)[1]
        for name, k in (('plain', K), ('comb', K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5))):
            pre.clear()
            r[f'r{n} {name} one'] = top1(atm_emb(m, x - k + K, fold, att)[0], g)
            f = torch.cat(pre) - mu
            r[f'r{n} {name} three'] = acc(sum(nz((f @ w + b).float(), dim=1) @ Gte[j].float().T for (w, b), j in zip(W, (0, 2, 3)))).item()
    rows.append(r); print({k: round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k40_self_key_reps_{TN}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
