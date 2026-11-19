"""K44: which key should the readout get, chosen inside the trial? K05: with the subject's true key known, the convex
combination of training keys closest to it beats their plain mean K (43.3 against 41.8). Pure inductive version: the
weights are fitted per trial on the 200 ms-periodic part (the in-trial key k of K36),
    w(x) = argmin over the simplex | comb(x) - comb(sum_j w_j m_j) |^2,   readout fed  x - k + [K + a (sum_j w_j m_j - K)]
a = 0 is K37 (comb), a = 1 the in-trial mixture key, a = 0.5 halfway. Frozen ATM trunk, attention of the raw trial, the
model's own head and the three distilled heads of K21, no re-centring.
Usage: k44_self_mixture.py [trunk ...]"""
import pandas as pd
from sk_lib import *

comb = lambda d: (lambda p: (p - p.mean(-1, keepdim=True)).repeat(*[1] * (d.dim() - 1), 5))(d.unflatten(-1, (5, 50)).mean(-2))
rows = []
for TRUNK in sys.argv[1:] or ['atm128_vc/seed3300']:
    TN = TRUNK.replace('/', '_')
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
        g, x = img_emb(m), T(eeg(fold))
        keys = torch.stack([train_key(s) for s in m.train_subs]); K = keys.mean(0, keepdim=True)
        k = K + comb(x - K)
        ck, cx = comb(keys).flatten(1), comb(x).flatten(1)  # (9, D), (200, D)
        w = torch.zeros(200, 9, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(300):
            opt.zero_grad(); ((torch.softmax(w, 1) @ ck - cx) ** 2).mean(1).sum().backward(); opt.step()
        w = torch.softmax(w, 1).detach()
        mix = (w @ keys.flatten(1)).view_as(x)
        att = atm_emb(m, x, fold)[1]
        r = dict(trunk=TRUNK, fold=fold, w_max=w.max(1).values.mean().item())
        for a in (0, 0.5, 1):
            pre.clear()
            r[f'a={a} own head'] = top1(atm_emb(m, x - k + K + a * (mix - K), fold, att)[0], g)
            f = torch.cat(pre) - mu
            r[f'a={a} three heads'] = acc(sum(nz((f @ wk + b).float(), dim=1) @ Gte[j].float().T for (wk, b), j in zip(W, (0, 2, 3)))).item()
        rows.append(r); print({k: v if k == 'trunk' else round(v, 2) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k44_self_mixture.csv', index=False)
print(df.groupby('trunk', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
