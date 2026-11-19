"""K41: fit the three distilled heads in the condition they are used in. K37 feeds the readout x - k + K (k = in-trial
key), but the heads of K21 were fitted on trunk features of training trials under their OWN key. Here the ridge heads
(teacher t6, members 0, 2, 3, same targets) are fitted on the trunk features of the training trials fed
  own    x                    (K21 / K37)
  K      x - m_s + K          (the training subject's true key replaced by the mean training key)
  comb   x - k(x) + K         (its in-trial key replaced: exactly the test-time operation)
always with the attention map of the raw trial. Test: held-out subject, plain / comb (pure inductive) / oracle key (test
data required), no re-centring; one head (L28) and three heads.
Usage: k41_heads_refit.py <folds, e.g. 1,5,7> [trunk]"""
import pandas as pd
from sk_lib import *

FOLDS = [int(f) for f in sys.argv[1].split(',')]
TRUNK = sys.argv[2] if len(sys.argv) > 2 else 'atm128_vc/seed3300'
comb = lambda d: (lambda p: (p - p.mean(-1, keepdim=True)).repeat(1, 1, 5))(d.unflatten(-1, (5, 50)).mean(-2))
rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    m = load(TRUNK, fold)
    pre = []
    m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.append(i[0].detach().cpu()))
    K = mean_key(m)

    def feats(x, s, keys):
        """Trunk features of trials x of subject s with the readout fed x - k + K, for each k in keys (a function of the trials)."""
        out = [[] for _ in keys]
        for i in range(0, len(x), 1000):
            xb = T(x[i:i + 1000])
            att = atm_emb(m, xb, s)[1]
            for o, key in zip(out, keys):
                pre.clear(); atm_emb(m, xb - key(xb) + K, s, att); o.append(torch.cat(pre).double())
        return [torch.cat(o) for o in out]

    fits = ('own', 'K', 'comb')
    tr = [feats(eeg(s, 'train'), s, (lambda x: K, lambda x, s=s: train_key(s)[None], lambda x: K + comb(x - K))) for s in t['subjects']]
    x = T(eeg(fold))
    tests = dict(zip(('plain', 'comb', 'oracle'), feats(x, fold, (lambda x: K, lambda x: K + comb(x - K), lambda x: x.mean(0, keepdim=True)))))
    r = dict(fold=fold)
    for j, fit in enumerate(fits):
        X = torch.cat([f[j] for f in tr]); mu = X.mean(0); X -= mu
        XtX = X.T @ X
        A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
        W = [(A @ (X.T @ (t['eeg'][k].double() - t['eeg'][k].double().mean(0))), t['eeg'][k].double().mean(0)) for k in (0, 2, 3)]
        for name, f in tests.items():
            S = [nz(((f - mu) @ w + b).float(), dim=1) @ Gte[k].float().T for (w, b), k in zip(W, (0, 2, 3))]
            r[f'fit {fit} / test {name}: one'], r[f'fit {fit} / test {name}: three'] = acc(S[0]).item(), acc(sum(S)).item()
        del X, XtX, A
    rows.append(r); print({k: round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k41_heads_refit_{TRUNK.replace("/", "_")}_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
