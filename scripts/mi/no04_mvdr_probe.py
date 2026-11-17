"""O04 per-trial adaptive noise filtering, read with a linear decoder before any retraining.
A trial is almost all noise, so its own channel covariance C_i (32 x 32, from its 120 samples) is its noise covariance.
The optimal read-out of a fixed topography a under that noise is a^T C_i^-1 x: a FIXED decoder applied to C_i^-1 x. A
network cannot invert a covariance by itself, so the transform would be a new input: x_hat = ((1-alpha) C_i + alpha C_pop)^-1 x
(shrunk toward the population covariance; alpha = 1 is a static transform, the control). Second form: C_pop C_s^-1 x, which is
x itself at alpha = 1, so it compares directly with the raw input.
Probe: one ridge decoder (3,840 cells -> first 64 principal axes of the training embeddings) per input variant, trained on
single trials of 8 Alljoined training people x 3,000 trained images, tested on the fold's NEW people x 2,500 never-trained
images, single trials and r=3, top-5 among the 5,000 never-trained images (in the 64-axis space), and the mean single-trial
cosine to the target (the low-noise read: top-5 counts are ~100 hits here), best ridge per variant.
    no04_mvdr_probe.py <fold>"""
from nc_lib import *

k = int(sys.argv[1]); NEW = FOLDS[k]; KNOWN = [s for s in range(1, 21) if s not in NEW][:8]
tr_img = np.setdiff1d(np.arange(16540), HELD)[::4][:3000]
G = torch.tensor(np.asarray(TARGETS)).float(); G = nz(G); mu = G[tr_img].mean(0)
PC = torch.linalg.svd(G[tr_img] - mu, full_matrices=False)[2][:64].T; Z = (G - mu) @ PC  # (16540, 64)
Xtr, _, itr = cell(KNOWN, tr_img); Xte, ste, ite = cell(NEW, HELD[:2500])
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte); Ttr = Z[tr_img][itr]
Cpop = torch.einsum('nct,ndt->cd', Xtr[::10], Xtr[::10]) / (len(Xtr[::10]) * 120)
gal = nz(Z[HELD]).to(DEV)


zs = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
PSD = lambda x: torch.nn.functional.avg_pool1d(torch.fft.rfft(x, dim=-1).abs().pow(2).flatten(0, 1)[:, None], 5, 1, 2, count_include_pad=False)[:, 0].view(*x.shape[:2], -1)
Ppop = PSD(Xtr[::10]).mean(0)  # (32, 61) population noise spectrum per channel


def spatial(x, a):  # D C_s^-1 x, shrunk toward the trial's own diagonal
    C = torch.einsum('nct,ndt->ncd', x, x) / 120; D = torch.diag_embed(torch.diagonal(C, dim1=1, dim2=2))
    return D @ torch.linalg.solve((1 - a) * C + a * D + 1e-6 * torch.eye(32, device=x.device), x)


def temporal(x, a):  # per trial and channel: spectrum divided by its own smoothed power, shrunk toward the population spectrum (identity at a = 1)
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * (Ppop.to(x.device) / ((1 - a) * PSD(x) + a * Ppop.to(x.device))).sqrt(), n=120, dim=-1)


def winsor(x, kk):  # clip every channel at kk robust sigmas of the trial (median absolute deviation)
    med = x.median(-1, keepdim=True).values; mad = (x - med).abs().median(-1, keepdim=True).values * 1.4826
    return torch.minimum(torch.maximum(x, med - kk * mad), med + kk * mad)


def spatial_knn(x, a, kk, w):  # the trial's covariance mixed (weight w) with the mean covariance of its kk nearest trials in its 64-batch
    C = torch.einsum('nct,ndt->ncd', x, x) / 120; Cn = C / torch.diagonal(C, dim1=1, dim2=2).sum(1).clamp_min(1e-9)[:, None, None]; out = []
    for b in range(0, len(x), 64):
        c = Cn[b:b + 64]; nn_ = torch.cdist(c.flatten(1), c.flatten(1)).topk(min(kk + 1, len(c)), largest=False).indices  # includes itself
        Cb = (1 - w) * C[b:b + 64] + w * (c[nn_].mean(1) * torch.diagonal(C[b:b + 64], dim1=1, dim2=2).sum(1)[:, None, None])
        D = torch.diag_embed(torch.diagonal(Cb, dim1=1, dim2=2))
        out.append(D @ torch.linalg.solve((1 - a) * Cb + a * D + 1e-6 * torch.eye(32, device=x.device), x[b:b + 64]))
    return torch.cat(out)


CHAIN = {'spatial .6 + z': lambda x: zs(spatial(x, .6)), 'temporal .6 + z': lambda x: zs(temporal(x, .6)), 'temporal .3 + z': lambda x: zs(temporal(x, .3)),
         'spatial .6 + temporal .6 + z': lambda x: zs(temporal(spatial(x, .6), .6)), 'temporal .6 + spatial .6 + z': lambda x: zs(spatial(temporal(x, .6), .6)),
         'winsor 3 + z': lambda x: zs(winsor(x, 3.)), 'winsor 3 + spatial .6 + z': lambda x: zs(spatial(winsor(x, 3.), .6)), 'spatial .6 + winsor 3 + z': lambda x: zs(winsor(spatial(x, .6), 3.)),
         'spatial .8 + z': lambda x: zs(spatial(x, .8)), 'knn5 w.5, .6 + z': lambda x: zs(spatial_knn(x, .6, 5, .5)), 'knn5 w.5, .4 + z': lambda x: zs(spatial_knn(x, .4, 5, .5)),
         'knn5 w1, .4 + z': lambda x: zs(spatial_knn(x, .4, 5, 1.)), 'knn5 w.5, .2 + z': lambda x: zs(spatial_knn(x, .2, 5, .5)), 'knn10 w.5, .4 + z': lambda x: zs(spatial_knn(x, .4, 10, .5))}


def variant(X, alpha):
    if alpha is None:
        return X.flatten(1)
    if alpha in CHAIN:
        return torch.cat([CHAIN[alpha](X[b:b + 8192].to(DEV)).flatten(1).cpu() for b in range(0, len(X), 8192)])
    if alpha == 'diag':  # per-trial, per-channel standardisation over time (what rmix does first)
        return ((X - X.mean(-1, keepdim=True)) / X.std(-1, keepdim=True).clamp_min(1e-4)).flatten(1)
    out = []
    for b in range(0, len(X), 8192):
        x = X[b:b + 8192].to(DEV); C = torch.einsum('nct,ndt->ncd', x, x) / 120
        if isinstance(alpha, tuple):  # self-contained: shrink toward the trial's own diagonal, D C_s^-1 x (identity at alpha = 1), optionally z-scored
            D = torch.diag_embed(torch.diagonal(C, dim1=1, dim2=2)); xh = D @ torch.linalg.solve((1 - alpha[0]) * C + alpha[0] * D + 1e-6 * torch.eye(32, device=DEV), x)
            if alpha[1]:
                xh = (xh - xh.mean(-1, keepdim=True)) / xh.std(-1, keepdim=True).clamp_min(1e-4)
            out.append(xh.flatten(1).cpu()); continue
        xh = torch.linalg.solve((1 - abs(alpha)) * C + abs(alpha) * Cpop.to(DEV), x)
        out.append((Cpop.to(DEV) @ xh if alpha < 0 else xh).flatten(1).cpu())  # alpha < 0: C_pop C_s^-1 x, the identity when the adaptation is off
    return torch.cat(out)


for name, alpha in ([(n, n) for n in CHAIN if os.environ['CHAIN'] in n] if 'CHAIN' in os.environ else []) + list((('raw x', None), ('per-trial channel z-score', 'diag'), ('self, alpha .8', (.8, 0)), ('self, alpha .6', (.6, 0)), ('self, alpha .4', (.4, 0)), ('self .6 + z-score', (.6, 1)), ('self .4 + z-score', (.4, 1)), ('C_pop C_s^-1 x, alpha .8', -.8), ('C_pop C_s^-1 x, alpha .6', -.6), ('C_pop C_s^-1 x, alpha .4', -.4), ('C_pop C_s^-1 x, alpha .2', -.2),
                    ('static C_pop^-1 x (control)', 1.), ('C_s^-1 x, alpha .6', .6), ('C_s^-1 x, alpha .3', .3), ('C_s^-1 x, alpha .1', .1)))[:0 if 'CHAIN' in os.environ else int(os.environ.get('NV', 7))]:
    A, B = variant(Xtr, alpha), variant(Xte, alpha); m, s = A.mean(0), A.std(0).clamp_min(1e-6); A, B = (A - m) / s, (B - m) / s
    Gm = torch.zeros(3840, 3840, device=DEV); Xy = torch.zeros(3840, 64, device=DEV)
    for b in range(0, len(A), 16384):
        a = A[b:b + 16384].to(DEV); Gm += a.T @ a; Xy += a.T @ Ttr[b:b + 16384].to(DEV)
    e, V = torch.linalg.eigh(Gm); Xy = V.T @ Xy; best = None
    for lam in (1e3, 1e4, 1e5, 1e6):
        W = V @ (Xy / (e + lam)[:, None]); y = torch.cat([B[b:b + 16384].to(DEV) @ W for b in range(0, len(B), 16384)]).cpu()
        sc = score(y, ste, ite, gal); c = (nz(y) * gal[torch.as_tensor(ite)].cpu()).sum(1)  # single-trial cosine to the target (64 axes, centred): the continuous read
        sc['cos'] = c.mean().item(); sc['se'] = c.std().item() / len(c) ** .5
        if best is None or sc['cos'] > best[1]['cos']:
            best = (lam, sc)
    print('%-30s best ridge %g: target cosine %.5f +- %.5f | top-5 r1 %.3f  r3 %.3f  r4 %.3f' % (name, best[0], best[1]['cos'], best[1]['se'], best[1]['r1'], best[1]['r3'], best[1]['r4']), flush=True)
