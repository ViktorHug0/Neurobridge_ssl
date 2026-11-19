"""K26: the three-head readout (K21) under the calibration sources of K11 (CPU, from the caches of k21_heads.py).
Per head the subject mean is estimated on: its 200 queries (self), the queries of the other 100 images (xhalf),
20 / 50 other queries, or the queries seen so far (causal running mean). Also no calibration at all.
Usage: k26_heads_calib_source.py"""
import numpy as np, pandas as pd, torch

SK = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_key'
nz = torch.nn.functional.normalize
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
        E, G, anc = [], [], []
        for k in (0, 2, 3):
            Y = t['eeg'][k].double()
            E.append(nz((Xte @ (A @ (X.T @ (Y - Y.mean(0)))) + Y.mean(0)).float(), dim=1)); G.append(Gte[k].float()); anc.append(t['img'][k].float().mean(0))

        def hits(idx, means):  # queries idx, each head shifted by its own mean vector(s)
            S = sum(nz(e[idx] - m + a, dim=1) @ g.T for e, g, a, m in zip(E, G, anc, means))
            return (S.argmax(1) == idx).float()

        allq = torch.arange(200)
        r = dict(seed=seed, fold=fold, none=100 * (sum(e @ g.T for e, g in zip(E, G)).argmax(1) == allq).float().mean().item(),
                 self=100 * hits(allq, [e.mean(0) for e in E]).mean().item())
        gen = torch.Generator().manual_seed(fold)
        acc = {k: [] for k in ('xhalf', 'n20', 'n50', 'causal', 'causal>20')}
        for _ in range(5):
            p = torch.randperm(200, generator=gen)
            a, b = p[:100], p[100:]
            acc['xhalf'].append((hits(a, [e[b].mean(0) for e in E]).mean().item() + hits(b, [e[a].mean(0) for e in E]).mean().item()) / 2)
            for n in (20, 50):
                acc[f'n{n}'].append(hits(p[n:], [e[p[:n]].mean(0) for e in E]).mean().item())
            h = hits(p, [e[p].cumsum(0) / torch.arange(1, 201)[:, None] for e in E])
            acc['causal'].append(h.mean().item()); acc['causal>20'].append(h[20:].mean().item())
        r.update({k: 100 * np.mean(v) for k, v in acc.items()})
        rows.append(r)
    print(seed, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k26_heads_calib_source.csv', index=False)
print(df.groupby('seed').mean().drop(columns='fold').round(2).to_string())
print(df.drop(columns=['seed', 'fold']).mean().round(2).to_string())
