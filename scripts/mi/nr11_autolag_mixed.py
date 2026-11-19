"""R11 the alignment step when a batch mixes people of every kind, as the sealed batches do (11 people per batch).
nr07 tested batches made of ONE fold's five held-out people; the calibration then depends on the fold (the norm peaks
1.9 samples before the decoding optimum for the people of folds 1 and 3, 0.6 for those of fold 2, for every architecture:
nr10). Here a virtual batch = one 64-trial batch from each of folds 1, 2, 3 (15 people, 192 trials, each fold decoded by
its own models): ONE shift per virtual batch from the summed detector curves (each divided by its mean), parabola + 1.5,
applied to all three parts. Pooled top-5 % (5k r=3 / r=4, 14k r=3) as is against aligned, for a common delay d.
    nr11_autolag_mixed.py [bias, default 1.5]"""
from nc_lib import *

bias = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5; GATE = float(os.environ.get('GATE', 0)); DEAD = int(os.environ.get('DEAD', 0)); KS = [int(a) for a in os.environ.get('KS', '1,2,3').split(',')]; S = list(range(-5, 6)); t = np.arange(-8, 9); gals = {'5k': gallery(), '14k': gallery(big=True)}; F = {}
for k in KS:
    E = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r'); n = E.shape[1]; X, subj, img = cell(FOLDS[k]); subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); F[k] = (E, np.load(f'{OUT}/autolag_norm_f{k}_mi_rmix_ho5k_lag5.npy')[:n // 64], g4, torch.as_tensor(img)[g4[:, 0]], n)
nb = min(v[4] // 64 for v in F.values())


def hits(k, tb):  # fold k decoded with batch b at total shift tb[b]
    E, _, g4, tg, n = F[k]; y = torch.empty(n, 1536); tb = np.clip(np.concatenate([tb, np.full(n // 64 - len(tb), tb[-1])]), S[0], S[-1])
    for s in np.unique(tb):
        rows = (np.nonzero(tb == s)[0][:, None] * 64 + np.arange(64)).ravel(); y[rows] = torch.as_tensor(np.asarray(E[S.index(s)][rows], dtype=np.float32))
    return [(rank(y[g4[:, :3]].mean(1), tg, gals['5k']) < 5).cpu().numpy(), (rank(y[g4].mean(1), tg, gals['5k']) < 5).cpu().numpy(), (rank(y[g4[:, :3]].mean(1), tg, gals['14k']) < 5).cpu().numpy()]


def vertex(c, v):
    j = np.clip(v.argmax(1), 1, len(c) - 2); r = np.arange(len(v)); l, m, h = v[r, j - 1], v[r, j], v[r, j + 1]; cur = l - 2 * m + h
    return c[j] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1) * (c[1] - c[0]), 0.0)


pool = lambda H: [np.concatenate([H[k][m] for k in H]) for m in range(3)]
for d in (-4, -2, 0, 2, 4):
    ok = (np.abs(t - d) <= 8) & ((t - d) % 2 == 0); curve = sum(F[k][1][:nb, ok] / F[k][1][:nb, ok].mean(1, keepdims=True) for k in F); tb = np.round(np.clip(vertex(t[ok], curve), t[ok][0], t[ok][-1]) + bias).astype(int)
    keep = (np.abs(tb - d) < DEAD) | (curve.max(1) / curve.mean(1) - 1 < GATE); tb[keep] = d  # DEAD / GATE: batches left as they are
    A = {k: hits(k, np.full(nb, d)) for k in F}; B = {k: hits(k, tb) for k in F}; a, b = pool(A), pool(B)
    i = np.random.default_rng(0).integers(0, len(a[0]), (2000, len(a[0])))
    print('gate %.2f dead %d delay %+d | acted %.0f %% | residual %+.2f sd %.2f | pooled as is %s -> aligned %s (%s) | per fold 5k r=3: %s' % (GATE, DEAD, d, 100 * (tb != d).mean(), tb.mean(), tb.std(), ' / '.join('%.3f' % (100 * x.mean()) for x in a), ' / '.join('%.3f' % (100 * x.mean()) for x in b),
          ' / '.join('%+.1f +- %.1f %%' % (100 * (y.mean() / x.mean() - 1), 100 * (y[i].mean(1) / x[i].mean(1) - 1).std()) for x, y in zip(a, b)), ' '.join('f%d %+.1f %%' % (k, 100 * (B[k][0].mean() / A[k][0].mean() - 1)) for k in F)), flush=True)
