"""K37: the in-trial key (K36: k = K + the 200 ms-periodic part of x - K, without its temporal mean) with the three-head
readout. Pure inductive setting: no re-centring, nothing from the new subject but the trial decoded.
Frozen ATM trunk + the closed-form distilled heads of K21 (teacher t6, members 0, 2, 3), attention map of the raw trial,
readout fed x - k + K with k = K (plain) / the in-trial key (comb) / the subject's mean ERP over its test trials (oracle,
test data required). Reported: the model's own head, the distilled L28 head, the three heads (mean of the cosine matrices).
Usage: k37_self_key_heads.py [trunk ...]"""
import pandas as pd
from sk_lib import *

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
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        per = (x - K).view(200, 63, 5, 50).mean(2)
        att = atm_emb(m, x, fold)[1]
        r = dict(trunk=TRUNK, fold=fold)
        for name, k in (('plain', K), ('comb', K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)), ('oracle', x.mean(0, keepdim=True))):
            pre.clear()
            r[f'{name} own head'] = top1(atm_emb(m, x - k + K, fold, att)[0], g)
            f = torch.cat(pre) - mu
            S = [nz((f @ w + b).float(), dim=1) @ Gte[j].float().T for (w, b), j in zip(W, (0, 2, 3))]
            r[f'{name} L28 head'], r[f'{name} three heads'] = acc(S[0]).item(), acc(sum(S)).item()
        rows.append(r); print({k: v if k == 'trunk' else round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k37_self_key_heads.csv', index=False)
print(df.groupby('trunk', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
