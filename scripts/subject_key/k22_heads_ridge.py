"""K22: the closed-form heads of K21, ridge strength and target choices (CPU only, from the caches of k21_heads.py).
Heads = the three ATM teachers on different image spaces (members 0, 2, 3 of t6) unless stated.
Usage: k22_heads_ridge.py [folds]"""
import os, sys
import numpy as np, pandas as pd, torch

SK = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_key'
FOLDS = [int(f) for f in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 5, 7]
nz = torch.nn.functional.normalize
acc = lambda S: 100 * (S.argmax(1) == torch.arange(len(S))).float().mean().item()
rank = lambda S: (S > S.diag()[:, None]).sum(1).float().mean().item()
rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    c = torch.load(f'{SK}/cache/trunk_atm128_vc_seed3300_f{fold}.pt')
    X, Xte = c['tr'].double(), c['te'].double()
    mu = X.mean(0); X -= mu; Xte -= mu
    XtX = X.T @ X
    ev, U = torch.linalg.eigh(XtX)
    XtY = {k: X.T @ (t['eeg'][k].double() - t['eeg'][k].double().mean(0)) for k in (0, 2, 3)}
    r = dict(fold=fold)
    for lam in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        A = (U / (ev + lam * ev.mean())) @ U.T
        S = []
        for k in (0, 2, 3):
            e = nz((Xte @ (A @ XtY[k]) + t['eeg'][k].double().mean(0)).float(), dim=1)
            S.append(nz(e - e.mean(0) + t['img'][k].float().mean(0), dim=1) @ Gte[k].float().T)
        r[f'lam{lam}'], r[f'lam{lam}_rank'] = acc(sum(S)), rank(sum(S))
    rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
print(df.mean().round(2).to_string())
