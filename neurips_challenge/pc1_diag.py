"""What is the top principal direction of a 64-trial batch? Stability across batches/subjects, relation to the
batch mean, to the trial norm, and whether a STATIC direction (mean PC1 over other subjects) works as well.
    pc1_diag.py <recipe>"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402

data = load([sys.argv[1]]); rng = np.random.default_rng(1)
pcs, res = {}, {}
for s, e in data.items():
    x = e['emb'][0]; o = rng.permutation(len(x)); v1 = []; cm = []; cn = []
    for i in range(0, len(o), 64):
        blk = x[o[i:i + 64]]; mu = blk.mean(0); c = blk - mu
        U, S, Vt = np.linalg.svd(c, full_matrices=False); v = Vt[0]
        v1.append(v); cm.append(abs(v @ mu) / np.linalg.norm(mu))
        cn.append(abs(np.corrcoef(c @ v, np.linalg.norm(blk, axis=1))[0, 1]))
    V = np.array(v1); pcs[s] = V
    G = np.abs(V @ V.T); np.fill_diagonal(G, np.nan)
    print(f's{s}: |cos| PC1 between batches {np.nanmean(G):.2f}; |cos(PC1, batch mean)| {np.mean(cm):.2f}; '
          f'|corr(PC1 score, trial norm)| {np.mean(cn):.2f}; var share PC1 {S[0]**2/(S**2).sum():.2f}', flush=True)
# static direction: leading eigenvector of the stacked PC1s of OTHER subjects
for k, subs in FOLDS.items():
    for s in subs:
        others = np.concatenate([pcs[t] for t in data if t != s])
        w = np.linalg.svd(others, full_matrices=False)[2][0]
        x = data[s]['emb'][0]; o = rng.permutation(len(x)); y = np.empty_like(x)
        for i in range(0, len(o), 64):
            idx = o[i:i + 64]; blk = x[idx]; mu = blk.mean(0); c = blk - mu
            y[idx] = c - 0.5 * np.outer(c @ w, w) + 0.5 * mu
        res.setdefault('static dir', []).append(score(dict(data[s], emb=[y]), np.ones((1, len(x)))))
        y2 = np.empty_like(x)
        for i in range(0, len(o), 64):
            idx = o[i:i + 64]; blk = x[idx]; mu = blk.mean(0); c = blk - mu
            v = np.linalg.svd(c, full_matrices=False)[2][0]
            y2[idx] = c - 0.5 * np.outer(c @ v, v) + 0.5 * mu
        res.setdefault('batch dir', []).append(score(dict(data[s], emb=[y2]), np.ones((1, len(x)))))
print({k: round(float(np.mean(v)), 2) for k, v in res.items()})
