"""Usage: .venv/bin/python neurips_challenge/affinity_labelfree.py (after affinity_summary.py; ~4 GB RAM, run via sbatch)."""
# Does label-free similarity between subjects' EEG predict the affinity interaction I[i,j] (source i -> target j)?
import sys; sys.path.insert(0, 'neurips_challenge')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from affinity_summary import load, fit, double_centre
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
rng = np.random.default_rng(0)
feats = {}
for s in range(1, 21):
    full = np.load(f'{D}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)  # (200 images x 80 repeats) trials
    full = full[np.abs(np.asarray(full[:, 0, :])).sum(1) > 0]                               # drop zero-padded repeats
    x = np.asarray(full[np.sort(rng.choice(len(full), 3000, replace=False))], dtype=np.float64)
    x = x - x.mean(-1, keepdims=True)
    C = np.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[2]); w, V = np.linalg.eigh(C)
    logC = (V * np.log(np.maximum(w, 1e-12))) @ V.T                                   # log-Euclidean covariance
    Cn = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))                                # channel correlation (scale-free)
    P = np.log(np.abs(np.fft.rfft(x, axis=-1)) ** 2 + 1e-12).mean(0)                 # (32, 61) log spectrum
    erp = np.asarray(full[::4], np.float64).mean(0)  # label-free grand-average ERP
    feats[s] = dict(logcov=logC.ravel(), chancorr=Cn[np.triu_indices(32, 1)], spectrum=(P - P.mean()).ravel(),
                    spec_shape=(P - P.mean(1, keepdims=True)).ravel(), erp=erp.ravel(), power=np.log(np.diag(C)))
subsets, scores = load()
a = fit(subsets, scores, range(40)); I = double_centre(a)[1:, 1:]; A = a[1:, 1:]
off = ~np.eye(20, dtype=bool)
def dc(M):
    M = M.copy(); M[~off] = np.nan; return double_centre(np.pad(M, ((1, 0), (1, 0)), constant_values=np.nan))[1:, 1:]
def mantel(S, T, n=2000):
    obs = np.corrcoef(S[off], T[off])[0, 1]; null = []
    for _ in range(n):
        p = rng.permutation(20); null.append(np.corrcoef(S[np.ix_(p, p)][off], T[off])[0, 1])
    return obs, (np.abs(null) >= abs(obs)).mean()
print('ceiling: interaction split-half 0.31 -> full-data reliability ~0.47, max observable r ~0.69')
for k in feats[1]:
    F = np.array([feats[s][k] for s in range(1, 21)])
    if k in ('chancorr', 'erp', 'spec_shape'):
        Fc = F - F.mean(1, keepdims=True); Fc /= np.linalg.norm(Fc, axis=1, keepdims=True); S = Fc @ Fc.T       # correlation similarity
    else:
        S = -np.linalg.norm(F[:, None] - F[None], axis=-1)                                                  # negative distance
    r1, p1 = mantel(dc(S), I); r2, p2 = mantel(S, A)
    print(f'{k:10s}  vs interaction r {r1:+.2f} (p {p1:.3f})   vs raw a r {r2:+.2f} (p {p2:.3f})')
