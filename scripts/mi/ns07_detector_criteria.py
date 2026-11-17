"""S07 a detector criterion with less person bias. S06: with batches of ONE person the packaged criterion (mean output norm
of the lag-only rmix over the batch) puts atypical people 2-4 samples away from their decoding optimum (which is 0..+1 for
everyone, O05 / R01): the norm follows the person's steady-state response shape, not only the image-locked response.
Criteria computed from the detector outputs y of a batch at each candidate shift (-8..+8 step 2), estimate = parabola vertex:
  norm     mean |y_i|                       (packaged)
  centred  mean |y_i - mean_batch(y)|       (the image-independent output removed)
  common   |mean_batch(y)|
  spread   1 - |mean_batch(y_i / |y_i|)|    (how much the outputs of the batch differ in direction)
  agree    mean cos(y_i, z_i), z = a second detector of another architecture (SQF-X lag-only)
  agree-c  the same after removing each detector's batch mean: agreement on the trial-specific part only
Per criterion: mixed 64-batches (mean, sd per batch) and one-person 64-batches (mean per person; sd across people = person bias).
Decoding (cached sub16-composition outputs, r=3, 5k, zero delay) with the shift each criterion applies, after a per-fold
constant that centres its mixed-batch mean on +0.75 (the decoding optimum): isolates the cost of the estimator's scatter.
    ns07_detector_criteria.py <fold> [detector recipe:checkpoint] [second detector recipe:checkpoint]"""
from nc_lib import *

k = int(sys.argv[1]); rec, ck = (sys.argv[2] if len(sys.argv) > 2 else 'mi_rmix_ho5k_lag5:checkpoint_ep10.pth').split(':'); S = list(range(-5, 6)); C = np.arange(-8, 9, 2); tt = torch.arange(120)
E = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r'); n = E.shape[1]; X, subj, img = cell(FOLDS[k]); X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery()
rec2, ck2 = (sys.argv[3] if len(sys.argv) > 3 else 'mi_sqfx_ho5k_lag5:checkpoint_test_best.pth').split(':'); Y = []
for r_, c_ in ((rec, ck), (rec2, ck2)):
    f = load(r_, k, c_); Y.append(np.stack([f(X[:, :, (tt - int(c)).clamp(0, 119)], bs=1000).numpy().astype(np.float16) for c in C])); del f; torch.cuda.empty_cache()  # (9, n, 1536)
Y, Z = Y
groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
per = [np.nonzero(subj == q)[0] for q in FOLDS[k]]; order = np.arange(n)
B = {'mixed': [order[i:i + 64] for i in range(0, n, 64)], 'one person': [q[i:i + 64] for q in per for i in range(0, len(q) - 31, 64)]}
cos = torch.nn.functional.cosine_similarity; ctr = lambda y: y - y.mean(1, keepdim=True)
CR = {'norm': lambda y, z: y.norm(dim=2).mean(1), 'centred': lambda y, z: ctr(y).norm(dim=2).mean(1), 'common': lambda y, z: y.mean(1).norm(dim=1),
      'spread': lambda y, z: 1 - (y / y.norm(dim=2, keepdim=True)).mean(1).norm(dim=1), 'norm2': lambda y, z: z.norm(dim=2).mean(1),
      'agree': lambda y, z: cos(y, z, dim=2).mean(1), 'agree-c': lambda y, z: cos(ctr(y), ctr(z), dim=2).mean(1)}


def vertex(v):  # (batches, 9) -> fractional estimate on the grid C
    j = np.clip(v.argmax(1), 1, len(C) - 2); r = np.arange(len(v)); l, m, h = v[r, j - 1], v[r, j], v[r, j + 1]; cur = l - 2 * m + h
    return np.clip(C[j] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1) * 2, 0.0), C[0], C[-1])


def ranks(tb):
    y = torch.empty(n, 1536)
    for s in np.unique(tb):
        rows = np.nonzero(tb == s)[0]; y[rows] = torch.as_tensor(np.asarray(E[S.index(s)], dtype=np.float32)[rows])
    return rank(y[g4[:, :3]].mean(1), tg, gal).numpy()


V = {}
for bn, bs in B.items():
    cur = {c: [] for c in CR}
    for b in bs:
        y, z = torch.as_tensor(Y[:, b].astype(np.float32)).to(DEV), torch.as_tensor(Z[:, b].astype(np.float32)).to(DEV)
        for c, fn in CR.items():
            cur[c].append(fn(y, z).cpu().numpy())
    V[bn] = {c: vertex(np.stack(v)) for c, v in cur.items()}
base = ranks(np.zeros(n, int)); bi = np.random.default_rng(0).integers(0, len(base), (1000, len(base))); who = np.array([subj[b[0]] for b in B['one person']])
print('fold %d detector %s | no alignment: top-5 %.3f %%' % (k, rec, 100 * (base < 5).mean()))
for c in CR:
    m, o = V['mixed'][c], V['one person'][c]; pm = [o[who == q].mean() for q in FOLDS[k]]; bias = 0.75 - m.mean(); line = []
    for bn, bs in B.items():
        tb = np.zeros(n, int)
        for b, v in zip(bs, V[bn][c]):
            tb[b] = int(round(v + bias))
        r = ranks(np.clip(tb, S[0], S[-1])); out = []
        for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
            a, d = fn(r).astype(float), fn(base).astype(float); out.append('%+5.1f +- %.1f' % (100 * (a.mean() / d.mean() - 1), 100 * (a[bi].mean(1) / d[bi].mean(1) - 1).std()))
        line.append('%s: top-5 %s, top-100 %s, ranking error %s' % (bn, *out))
    print('  %-8s mixed: estimate %+.2f sd %.2f | one person: per person %s (sd across people %.2f, per-batch sd within a person %.2f) | decoding vs none, %s' % (
        c, m.mean(), m.std(), ' '.join('%+.1f' % v for v in pm), np.std(pm), np.mean([o[who == q].std() for q in FOLDS[k]]), ' || '.join(line)), flush=True)
