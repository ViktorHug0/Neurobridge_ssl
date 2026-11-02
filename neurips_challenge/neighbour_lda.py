"""Label-free in-batch routing with a LEARNED subject metric. Fingerprint per trial (raw EEG, no labels): per-channel
log-variance, channel correlations, per-channel log band power (4 bands). An LDA projection that separates
participants is fitted on the OTHER folds' 15 subjects (participant identity of training subjects is known offline;
the evaluated fold's subjects are unseen). In each shuffled mixed 64-batch, each trial is centred on the mean of
its K nearest batch-mates in LDA space, then the top batch direction is shrunk.
Also reports in-batch neighbour purity (fraction of the K neighbours that are the same participant).
    neighbour_lda.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402

D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]


def fingerprint(s, n_keep=None, rng=None):
    x = np.load(f'{D}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
    out = []
    for i in range(0, len(x), 2000):
        b = np.asarray(x[i:i + 2000], np.float32); b = b - b.mean(-1, keepdims=True)
        v = (b ** 2).mean(-1) + 1e-6
        c = np.einsum('nct,ndt->ncd', b, b) / 120 / np.sqrt(v[:, :, None] * v[:, None, :])
        iu = np.triu_indices(32, 1)
        P = np.abs(np.fft.rfft(b, axis=-1)) ** 2
        bp = np.concatenate([np.log(P[..., lo:hi].mean(-1) + 1e-6) for lo, hi in BANDS], 1)
        out.append(np.hstack([np.log(v), c[:, iu[0], iu[1]], bp]))
    return np.concatenate(out)


def lda(F, y, dim=32, reg=1e-2):
    mu = F.mean(0); classes = np.unique(y)
    Sw = sum(np.cov(F[y == c], rowvar=False) for c in classes) / len(classes)
    M = np.stack([F[y == c].mean(0) for c in classes]) - mu
    Sb = M.T @ M / len(classes)
    Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    w, V = np.linalg.eigh(Sw); Wi = (V / np.sqrt(w)) @ V.T                 # Sw^-1/2
    e, U = np.linalg.eigh(Wi @ Sb @ Wi)
    return mu, Wi @ U[:, ::-1][:, :dim]


def tf(q, f, K, a=.5, b=.5):
    if K:
        Dm = ((f[:, None] - f[None]) ** 2).sum(-1)
        nb = np.argsort(Dm, 1)[:, :K]; mu = q[nb].mean(1)
    else:
        mu = np.repeat(q.mean(0)[None], len(q), 0)
    c = q - mu
    V = np.linalg.svd(c, full_matrices=False)[2][:1]
    return c - b * (c @ V.T) @ V + (1 - a) * mu


def main():
    recipes = sys.argv[1:]; data = load(recipes); M = len(recipes); rng = np.random.default_rng(6)
    FP = {s: fingerprint(s) for s in range(1, 21)}
    res, pur = {}, {}
    for kf, subs in FOLDS.items():
        train = [s for s in FP if s not in subs]
        Ftr = np.concatenate([FP[s][rng.choice(len(FP[s]), 3000, replace=False)] for s in train])
        ytr = np.repeat(train, 3000)
        sd = Ftr.std(0) + 1e-6; m0 = Ftr.mean(0)
        mu, W = lda((Ftr - m0) / sd, ytr)
        n = len(data[subs[0]]['label'])
        F = np.concatenate([FP[s] for s in subs]); owner = np.repeat(np.arange(len(subs)), n)
        spaces = {'raw z': (F - F.mean(0)) / (F.std(0) + 1e-6), 'LDA': ((F - m0) / sd - mu) @ W}
        order = rng.permutation(len(F))
        for sname, Fs in spaces.items():
            for K in (0, 4, 8, 16) if sname == 'LDA' else (4, 8):
                outs = []
                for m in range(M):
                    allx = np.concatenate([data[s]['emb'][m] for s in subs]); y = np.empty_like(allx)
                    for j in range(0, len(order), 64):
                        idx = order[j:j + 64]
                        y[idx] = tf(allx[idx], Fs[idx], K)
                        if m == 0 and K:
                            Dm = ((Fs[idx][:, None] - Fs[idx][None]) ** 2).sum(-1)
                            nb = np.argsort(Dm, 1)[:, 1:K + 1]
                            pur.setdefault(f'{sname} K={K}', []).append((owner[idx][nb] == owner[idx][:, None]).mean())
                    outs.append(y)
                for i, s in enumerate(subs):
                    res.setdefault(f'{sname} K={K}', {})[s] = score(dict(data[s], emb=[o[i * n:(i + 1) * n] for o in outs]), np.ones((M, n)))
        for s in subs:
            res.setdefault('plain', {})[s] = score(data[s], np.ones((M, len(data[s]['label']))))
    subs = sorted(data); base = np.array([res['plain'][s] for s in subs])
    print('+'.join(recipes), '(shuffled mixed 64-batches of 5 unseen subjects; chance neighbour purity ~0.19)')
    for name, v in res.items():
        v = np.array([v[s] for s in subs])
        p = f'purity {np.mean(pur[name]):.2f}' if name in pur else ''
        print(f'  {name:14s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}  {p}')


if __name__ == '__main__':
    main()
