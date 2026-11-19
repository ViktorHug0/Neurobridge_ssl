"""K34: few-shot output rotation (K33) on the three-head readout (K21). Frozen ATM trunk + the closed-form distilled
heads (teacher t6, members 0, 2, 3). For each head, R = orthogonal Procrustes with identity prior (n0 = 1000) between the
head's embeddings of n LABELLED training-image trials of the new subject and that space's image embeddings; test queries
re-centred with their own mean, rotated, moved to the image anchor; cosine matrices of the heads summed.
n = 0 is re-centring alone. 3 draws per n.
Usage: k34_heads_rotation.py [trunk]"""
import pandas as pd
from sk_lib import *
from module.eeg_encoder.model import run_eeg_backbone

TRUNK = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
TN = TRUNK.replace('/', '_')


def rotation(X, Y, n0):  # as in K33
    X, Y = X - X.mean(0), Y - Y.mean(0)
    C = X.T @ Y
    U, _, Vt = torch.linalg.svd(C + n0 * C.trace() / (len(X) * len(C)) * torch.eye(len(C)))
    return U @ Vt


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

    @torch.no_grad()
    def feats(x):
        out = []
        for i in range(0, len(x), 100):
            run_eeg_backbone(m.enc, m.args, x[i:i + 100].to(DEV), torch.full((len(x[i:i + 100]),), fold, device=DEV)); out.append(pre['x'])
        return torch.cat(out) - mu

    fte, ftr = feats(T(eeg(fold))), feats(T(eeg(fold, 'train')))
    E = [(nz((fte @ w + b).float(), dim=1), nz((ftr @ w + b).float(), dim=1), t['img'][k].float(), Gte[k].float()) for (w, b), k in zip(W, (0, 2, 3))]
    r = dict(fold=fold)
    for n in (0, 50, 200, 1000, 4000, 16540):
        a = []
        for draw in range(3 if 0 < n < 16540 else 1):
            i = torch.randperm(16540, generator=torch.Generator().manual_seed(draw))[:n]
            S = [nz((e - e.mean(0)) @ (rotation(etr[i], G[i], 1000) if n else torch.eye(e.shape[1])) + G.mean(0), dim=1) @ g.T for e, etr, G, g in E]
            a.append((acc(S[0]).item(), acc(sum(S)).item()))
        r[f'n{n} single'], r[f'n{n} three'] = np.mean(a, 0)
    rows.append(r); print({k: round(float(v), 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k34_heads_rotation_{TN}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
