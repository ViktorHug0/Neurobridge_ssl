"""S12 the zero-delay price of the alignment for the ACTUAL package members (19 Nov, user: be critical about the alignment). S09 / S11
priced it on the cached decoding of the sub16 six, whose proxy runs are mostly without time jitter; E2's members are mostly jitter-trained.
Folds 1 + 3 (10 held-out people), batches of 64 mixing them, three batch seeds; the package's estimator on the detector norms
(`s06_trialnorm_f<k>.npy`); E2 outputs at total shifts -1, 0, +1, +2 (`u01s<shift>_...`; proposals outside are clipped and counted);
per-trial and r=3 top-5 (four triplets) against no alignment, for E2 and, on the same batches, for the six of S09.
    ns12_align_e2.py"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 3]; C = np.arange(-8, 9); SHIFTS = (-1, 0, 1, 2)
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
gal = gallery(); F = {}; per, nrm, fold = [], [], []
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; del X; subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)]))
    Y = {s: sum(w * scaled(torch.as_tensor(np.load(f"{OUT}/u01{'s%d' % s if s else ''}_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32))) for r, c, w in E2).half() for s in SHIFTS}
    N = np.load(f'{OUT}/s06_trialnorm_f{k}.npy'); F[k] = (Y, np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r'), torch.as_tensor(g4), g3, torch.as_tensor(img))
    per.append(subj); nrm.append(N / N.mean()); fold.append(np.full(n, k)); print('fold', k, 'loaded', flush=True)
per, nrm, fold = map(np.concatenate, (per, nrm, fold)); T = len(per)


def batches(seed):
    r = np.random.default_rng(seed); pools = {p: list(r.permutation(np.nonzero(per == p)[0])) for p in np.unique(per)}; out = []
    while any(pools.values()):
        b = []
        while len(b) < 64:
            cand = [p for p in pools if pools[p]]
            if not cand:
                break
            b.append(pools[cand[r.integers(len(cand))]].pop())
        if len(b) >= 32:
            out.append(np.array(b))
    return out


CAL = float(os.environ.get('CAL', 1.5)); DELAYS = [int(a) for a in os.environ.get('D', '0').split(',')]  # CAL: the estimator's bias (package aux 'bias'); D: uniform delays (-1..2)


def estimate(b, d=0):  # the package's estimator under a uniform delay d (ns11): (curve prominence, total shift after correction)
    ok = (np.abs(C - d) <= 8) & ((C - d) % 2 == 0); c = C[ok]; v = nrm[b][:, ok].mean(0); j = min(max(int(v.argmax()), 1), len(c) - 2); l, m, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m + h
    return v.max() / v.mean() - 1, int(round(min(max(c[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.), c[0]), c[-1]) + CAL))


def hits(s, which):  # per-trial total shift -> (per-trial hits by group, r=3 hits by group), folds concatenated
    h1, h3 = [], []
    for k in KS:
        Y, E, g4, g3, img = F[k]; sk = s[fold == k].astype(int); y = torch.empty(len(sk), 1536)
        for sh in np.unique(sk):
            m = np.nonzero(sk == sh)[0]; y[m] = Y[sh][m].float() if which == 'E2' else torch.as_tensor(np.asarray(E[sh + 5][m]).astype(np.float32))
        h1.append((rank(y[g4.ravel()], img[g4.ravel()], gal) < 5).float().reshape(-1, 4).mean(1).numpy()); h3.append((rank(y[g3].mean(1), img[g3[:, 0]], gal) < 5).float().reshape(4, -1).mean(0).numpy())
    return np.concatenate(h1), np.concatenate(h3)


for which in os.environ.get('WHICH', 'E2,six').split(','):
    z1, z3 = hits(np.zeros(T), which); print('%s, bias %.1f, no alignment at zero delay: per-trial %.3f %%, r=3 %.3f %%, gain %.3f' % (which, CAL, 100 * z1.mean(), 100 * z3.mean(), z3.mean() / z1.mean()), flush=True)
    for d in DELAYS:
        b1, b3 = (z1, z3) if d == 0 else hits(np.full(T, d), which)
        if d:
            print(' delay %+d uncorrected, vs zero delay: per-trial %+.1f ± %.1f, r=3 %+.1f ± %.1f' % (d, *paired(b1, z1, 300), *paired(b3, z3, 300)), flush=True)
        for lab, gate, dead in (('gate 0.10', .1, 0), ('gate + dead zone 2', .1, 2), ('gate + dead zone 3', .1, 3)):
            A1, A3, cors, clipped = [], [], [], 0
            for seed in range(3):
                s = np.full(T, d)
                for b in batches(seed):
                    pr, t = estimate(b, d); ok = pr >= gate and abs(t - d) >= dead; cors.append(t - d if ok else 0); clipped += ok and not -1 <= t <= 2
                    if ok:
                        s[b] = min(max(t, -1), 2)
                a1, a3 = hits(s, which); A1.append(a1); A3.append(a3)
            cors = np.array(cors); print('   delay %+d %-20s vs uncorrected: per-trial %+.2f ± %.2f | r=3 %+.2f ± %.2f | corrections %s | clipped %d of %d' % (
                d, lab, *paired(np.concatenate(A1), np.tile(b1, 3), 500), *paired(np.concatenate(A3), np.tile(b3, 3), 500), ' '.join('%+d:%.0f%%' % (c, 100 * (cors == c).mean()) for c in np.unique(cors)), clipped, len(cors)), flush=True)
