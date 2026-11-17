"""S05 matching the second moments of the input to the training population: 32 x 32 channel covariance (E03 re-rendering:
x -> C_pop^(1/2) C^(-1/2) x), its diagonal only, and the power spectrum per electrode. E03 found the covariance version
harmful for OUR new people (-8 %); a cohort recorded elsewhere is the case it could help. Statistics C / P taken over the
whole cohort (best case of a batch estimate), over each 64-batch, or over each person (needs grouping: an upper bound).
Population = the Alljoined people of fold 3 (training people of the fold-1 models). d6 + rmix + SQF-X (jitter) of fold 1, r=3.
    ns05_moment_match.py"""
from nc_lib import *

fr = torch.arange(61.)


def roots(X, shrink=0.05):  # as nq08: R^(1/2), R^(-1/2) of the channel covariance (on the CPU, in chunks)
    c = sum(torch.einsum('nct,ndt->cd', x.double(), x.double()) for x in X.split(8192)) / (X.shape[0] * X.shape[-1]); c = (1 - shrink) * c + shrink * torch.eye(len(c), dtype=torch.float64) * torch.diagonal(c).mean()
    e, v = torch.linalg.eigh(c); e = e.clamp_min(1e-8); return ((v * e.sqrt()) @ v.T).float(), ((v * e.rsqrt()) @ v.T).float()


psd = lambda X: sum(torch.fft.rfft(x, dim=-1).abs().pow(2).sum(0) for x in X.split(8192)) / len(X)  # (32, 61)
Xp = torch.as_tensor(cell(FOLDS[3])[0][:40000]); Rp = roots(Xp)[0]; Dp = Xp.pow(2).mean((0, 2)).sqrt(); Pp = psd(Xp)


def match(X, kind, idx):  # idx: list of row-index tensors sharing one statistic
    Y = X.clone()
    for i in idx:
        x = X[i]
        if kind == 'covariance':
            Y[i] = torch.einsum('cd,ndt->nct', Rp @ roots(x)[1], x)
        elif kind == 'variance':
            Y[i] = x * (Dp / x.pow(2).mean((0, 2)).sqrt())[:, None]
        else:
            Y[i] = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * (Pp / psd(x).clamp_min(1e-8)).sqrt().clamp(.5, 2.), n=120, dim=-1)
    return Y


data = {}
X, subj, img = cell(FOLDS[1]); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
data['fold1'] = (torch.as_tensor(X), subj, img, gallery(), torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])))
for c in ('ds004018', 'ds007964'):
    data[c] = cohort(c)
fs = [load(r, 1, c) for r, c in (('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'))]
for name, (X, subj, img, gal, g3) in data.items():
    n = len(X); lv = {'cohort': [torch.arange(n)], 'batch': list(torch.arange(n).split(64)), 'person': [torch.as_tensor(np.nonzero(subj == p)[0]) for p in sorted(set(subj.tolist()))]}
    tg = torch.as_tensor(img)[g3[:, 0]]; base = None; print(name, '| distance of the cohort covariance to the population (relative Frobenius): %.2f' % ((roots(X[:40000])[0] - Rp).norm() / Rp.norm()).item())
    for kind, level in [('identity', 'cohort')] + [(a, b) for a in ('covariance', 'variance', 'spectrum') for b in ('cohort', 'batch', 'person')]:
        Xt = X if kind == 'identity' else match(X, kind, lv[level]); r = rank(sum(scaled(f(Xt, bs=1000)) for f in fs)[g3].mean(1), tg, gal).numpy(); base = r if base is None else base
        bi = np.random.default_rng(0).integers(0, len(r), (500, len(r))); out = []
        for fn in (lambda v: v < 5, lambda v: v / (len(gal) - 1.)):
            p, q = fn(r).astype(float), fn(base).astype(float); out.append('%.3f (%+5.1f +- %.1f %%)' % (p.mean() * (100 if fn(np.array([0]))[0] == 1 else 1), 100 * (p.mean() / q.mean() - 1), 100 * (p[bi].mean(1) / q[bi].mean(1) - 1).std()))
        print('  %-11s per %-7s top-5 %s | ranking error %s' % (kind, level, *out), flush=True)
