"""K28: calibrating the three-head readout from very few unlabelled trials, with shrinkage of the estimated shift:
e <- e - n / (n + n0) * (mean of n calibration trials - anchor). CPU, caches of k21_heads.py, trunk seed 3300,
10 folds x 10 draws; the remaining queries are scored."""
import numpy as np, torch

SK = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_key'
nz = torch.nn.functional.normalize
res = {}
for fold in range(1, 11):
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu'); Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    c = torch.load(f'{SK}/cache/trunk_atm128_vc_seed3300_f{fold}.pt'); X, Xte = c['tr'].double(), c['te'].double()
    mu = X.mean(0); X -= mu; Xte -= mu
    XtX = X.T @ X
    A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
    E, G, anc = [], [], []
    for k in (0, 2, 3):
        Y = t['eeg'][k].double()
        E.append(nz((Xte @ (A @ (X.T @ (Y - Y.mean(0)))) + Y.mean(0)).float(), dim=1)); G.append(Gte[k].float()); anc.append(t['img'][k].float().mean(0))
    gen = torch.Generator().manual_seed(fold)
    for _ in range(10):
        p = torch.randperm(200, generator=gen)
        for n in (3, 5, 10, 20, 50):
            for n0 in (0, 2, 5, 10):
                idx = p[n:]
                S = sum(nz(e[idx] - n / (n + n0) * (e[p[:n]].mean(0) - a), dim=1) @ g.T for e, g, a in zip(E, G, anc))
                res.setdefault((n, n0), []).append((S.argmax(1) == idx).float().mean().item())
    S = sum(e @ g.T for e, g in zip(E, G))
    res.setdefault('none', []).append((S.argmax(1) == torch.arange(200)).float().mean().item())
print('no calibration', round(100 * np.mean(res['none']), 1))
for n in (3, 5, 10, 20, 50):
    print('n =', n, {f'n0={n0}': round(100 * np.mean(res[(n, n0)]), 1) for n0 in (0, 2, 5, 10)})
