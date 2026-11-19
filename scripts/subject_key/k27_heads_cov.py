"""K27: second-order calibration (K14) for the three-head readout of K21 (CPU, from the caches of k21_heads.py).
Per head: mean shift only (K10), or mean shift blended half-way with whitening by the subject's own covariance and
re-colouring by the covariance of that head's image embeddings of the training images (shrinkage 0.6).
Usage: k27_heads_cov.py"""
import pandas as pd, torch

SK = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_key'
nz = torch.nn.functional.normalize
acc = lambda S: 100 * (S.argmax(1) == torch.arange(len(S))).float().mean().item()
rank = lambda S: (S > S.diag()[:, None]).sum(1).float().mean().item()


def cov(a, s=0.6):
    c = torch.cov(a.T.double())
    return (1 - s) * c + s * torch.trace(c) / len(c) * torch.eye(len(c), dtype=torch.float64)


def power(c, p):
    w, v = torch.linalg.eigh(c)
    return (v * w.clamp_min(1e-8) ** p) @ v.T


rows = []
for seed in (3300, 3301, 3302):
    for fold in range(1, 11):
        t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
        Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
        c = torch.load(f'{SK}/cache/trunk_atm128_vc_seed{seed}_f{fold}.pt')
        X, Xte = c['tr'].double(), c['te'].double()
        mu = X.mean(0); X -= mu; Xte -= mu
        XtX = X.T @ X
        A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
        S1, S2, one = [], [], {}
        for k in (0, 2, 3):
            Y = t['eeg'][k].double(); g = t['img'][k].float()
            e = nz((Xte @ (A @ (X.T @ (Y - Y.mean(0)))) + Y.mean(0)).float(), dim=1)
            c0 = e - e.mean(0)
            W = (power(cov(e), -0.5) @ power(cov(g), 0.5)).float()
            S1.append(nz(c0 + g.mean(0), dim=1) @ Gte[k].float().T)
            S2.append(nz(0.5 * c0 + 0.5 * c0 @ W + g.mean(0), dim=1) @ Gte[k].float().T)
        rows.append(dict(seed=seed, fold=fold, own_mean=acc(S1[0]), own_cov=acc(S2[0]), heads_mean=acc(sum(S1)), heads_cov=acc(sum(S2)),
                         heads_mean_rank=rank(sum(S1)), heads_cov_rank=rank(sum(S2))))
    print(seed, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k27_heads_cov.csv', index=False)
print(df.groupby('seed').mean().drop(columns='fold').round(2).to_string())
print(df.drop(columns=['seed', 'fold']).mean().round(2).to_string())
