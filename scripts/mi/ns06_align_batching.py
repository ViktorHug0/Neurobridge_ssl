"""S06 the packaged alignment step under other batch compositions. Organisers (Discord, Oct): sealed batches "will be
shuffled and may mix participants"; the warm-up loader served subject after subject. nr07 / nr11 validated the step on
batches mixing people. Here: per-trial detector norms (lag-only rmix of the fold) at the candidate shifts -8..+8 step 2,
then any batching is an average of them: 'mixed' (the fold's 5 people shuffled), 'one person' (64 trials of one person),
and batch sizes 32 / 128. Estimator = the package's: parabola vertex + 1.5, rounded. Decoding with the applied shift from
the cached sub16-composition outputs (`autolag_e2e_f<k>.npy`, shifts -5..+5), r=3, 5k, against no alignment, at a
common delay d of the data (0, +-2, +-4: candidate grid moved by d).
GATE=<p>: the step acts only on batches whose curve peaks at least p above its mean over the 9 candidates (confidence gate).
    [GATE=p] [E2E=<tag>: the composition cached by nw01_e2e_cache.py] ns06_align_batching.py <fold>"""
from nc_lib import *

k = int(sys.argv[1]); GATE = float(os.environ.get('GATE', 0)); S = list(range(-5, 6)); C = np.arange(-8, 9); tt = torch.arange(120); E = np.load(f"{OUT}/autolag_e2e_{os.environ['E2E'] + '_' if os.environ.get('E2E') else ''}f{k}.npy", mmap_mode='r'); n = E.shape[1]
X, subj, img = cell(FOLDS[k]); X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); p = f'{OUT}/s06_trialnorm_f{k}.npy'
if not os.path.exists(p):
    f = load('mi_rmix_ho5k_lag5', k, 'checkpoint_ep10.pth'); np.save(p, np.stack([f(X[:, :, (tt - int(c)).clamp(0, 119)], bs=1000).norm(dim=1).numpy() for c in C], 1)); del f
N = np.load(p); groups = {}  # (n, 17): per-trial norm at every candidate shift
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]; s3 = subj[g4[:, 0].numpy()]


def applied(batches, d):  # per-trial total shift (delay d + correction) chosen by the package's estimator on each batch
    ok = (np.abs(C - d) <= 8) & ((C - d) % 2 == 0); c = C[ok]; out = np.full(n, d)
    for b in batches:
        if len(b) < 32:
            continue
        v = N[b][:, ok].mean(0); j = min(max(int(v.argmax()), 1), len(c) - 2); l, m, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m + h
        if v.max() / v.mean() - 1 < GATE:  # no clear peak over the candidate shifts: leave the batch as it is
            continue
        out[b] = int(round(min(max(c[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.), c[0]), c[-1]) + 1.5))
    return np.clip(out, S[0], S[-1])


def ranks(tb):
    y = torch.empty(n, 1536)
    for s in np.unique(tb):
        rows = np.nonzero(tb == s)[0]; y[rows] = torch.as_tensor(np.asarray(E[S.index(s)], dtype=np.float32)[rows])
    return rank(y[g4[:, :3]].mean(1), tg, gal).numpy()


order = np.arange(n); per = [np.nonzero(subj == q)[0] for q in FOLDS[k]]
B = {'mixed, 64': [order[i:i + 64] for i in range(0, n, 64)], 'mixed, 32': [order[i:i + 32] for i in range(0, n, 32)], 'mixed, 128': [order[i:i + 128] for i in range(0, n, 128)],
     'one person, 64': [q[i:i + 64] for q in per for i in range(0, len(q), 64)]}
for d in (0, -2, 2, -4, 4):
    base = ranks(np.full(n, d)); bi = np.random.default_rng(0).integers(0, len(base), (1000, len(base)))
    for name, bs in B.items():
        tb = applied(bs, d); r = ranks(tb); out = []
        for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
            a, c = fn(r).astype(float), fn(base).astype(float); out.append('%+5.1f +- %.1f' % (100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
        print('gate %.2f fold %d delay %+d | %-15s acted on %.0f %% of the trials | residual shift mean %+.2f sd %.2f | per person %s | vs no alignment: top-5 %s | top-100 %s | ranking error %s | per person top-5 %s' % (
            GATE, k, d, name, 100 * (tb != d).mean() if d else float('nan'), tb.mean(), tb.std(), ' '.join('%+.1f' % tb[q].mean() for q in per), *out, ' '.join('%+.0f%%' % (100 * ((r[s3 == q] < 5).mean() / max((base[s3 == q] < 5).mean(), 1e-9) - 1)) for q in FOLDS[k])), flush=True)
