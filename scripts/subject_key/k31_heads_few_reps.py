"""K31: the three-head readout when queries have few repetitions. Frozen ATM trunk + the closed-form distilled heads
of K21 (teacher t6, members 0, 2, 3). Query = mean of the first r repetitions, r = 1, 4, 16, 80; re-centring mean from
the same r-repetition queries. Single head (the L28 head) and three heads, plain and re-centred.
Usage: k31_heads_few_reps.py [trunk]"""
import pandas as pd
from sk_lib import *
from module.eeg_encoder.model import run_eeg_backbone

TRUNK = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
TN = TRUNK.replace('/', '_')
rows = []
for fold in range(1, 11):
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    c = torch.load(f'{SK}/cache/trunk_{TN}_f{fold}.pt')
    X = c['tr'].double(); mu = X.mean(0); X -= mu
    XtX = X.T @ X
    A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
    W = [(A @ (X.T @ (t['eeg'][k].double() - t['eeg'][k].double().mean(0))), t['eeg'][k].double().mean(0)) for k in (0, 2, 3)]
    m = load(TRUNK, fold)
    pre = {}
    m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.__setitem__('x', i[0].detach().double().cpu()))
    reps = np.asarray(eeg(fold, 'test', average=False))
    r = dict(fold=fold)
    for n in (1, 4, 16, 80):
        f = []
        for i in range(0, 200, 50):
            with torch.no_grad():
                run_eeg_backbone(m.enc, m.args, T(reps[i:i + 50, :n].mean(1)).to(DEV), torch.full((50,), fold, device=DEV))
            f.append(pre['x'])
        f = torch.cat(f) - mu
        P, S = [], []
        for (w, b), k in zip(W, (0, 2, 3)):
            e = nz((f @ w + b).float(), dim=1)
            P.append(e @ Gte[k].float().T)
            S.append(nz(e - e.mean(0) + t['img'][k].float().mean(0), dim=1) @ Gte[k].float().T)
        r.update({f'r{n} single': acc(P[0]).item(), f'r{n} single+cal': acc(S[0]).item(), f'r{n} three': acc(sum(P)).item(), f'r{n} three+cal': acc(sum(S)).item()})
    rows.append(r); print({k: round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k31_heads_few_reps_{TN}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
