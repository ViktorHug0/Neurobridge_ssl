"""S10 one shift per batch when the people of a batch carry DIFFERENT delays (user, 18 Nov: batches mixing cohorts with their own lags).
Held-out Alljoined people of the given folds (fold models: nothing in-sample), 64-trial batches mixed over all of them. Each person gets a
delay d_p; a batch's detector curve is the mean over its trials of their norm at total shift d_p + c (per-trial norms of the fold's
lag-only rmix, `s06_trialnorm_f<k>.npy`), the package's estimator picks one c*, a policy decides whether to apply it, and every trial is
decoded at d_p + c* (cached `autolag_e2e_f<k>.npy`, totals clipped to -5..+5). r=3, 5k gallery. Top-5 against no alignment, and the
same data with no delay at all (the ceiling).
    ns10_mixed_lag.py <folds, e.g. 1,2,3,4>"""
from nc_lib import *


def mix11(per, n_people=11, seed=0):  # 64-trial batches, each drawn from n_people people picked at random (the sealed cohort has 11)
    r = np.random.default_rng(seed); pools = {p: list(r.permutation(np.nonzero(per == p)[0])) for p in np.unique(per)}; out = []
    while any(pools.values()):
        alive = [p for p in pools if pools[p]]; pick = list(r.choice(alive, min(n_people, len(alive)), replace=False)); b = []
        while len(b) < 64:
            cand = [p for p in pick if pools[p]]
            if not cand:
                break
            b.append(pools[cand[r.integers(len(cand))]].pop())
        if len(b) >= 32:
            out.append(np.array(b))
    return out

KS = [int(a) for a in sys.argv[1].split(',')]; C = np.arange(-8, 9)
POL = [('Always', 0., 0, 99, False), ('Gate 0.10 (finalA)', .1, 0, 99, False), ('Gate + cap 5', .1, 0, 5, False), ('Gate + reject beyond 4', .1, 0, 4, True),
       ('Gate + reject beyond 3', .1, 0, 3, True), ('Dead zone 2', 0., 2, 99, False), ('Dead zone 3', 0., 3, 99, False)]
fold, per, nrm, G4 = [], [], [], {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r').shape[1]; N = np.load(f'{OUT}/s06_trialnorm_f{k}.npy'); subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); G4[k] = (g4, torch.as_tensor(img)[g4[:, 0]])
    fold.append(np.full(n, k)); per.append(subj); nrm.append(N / N.mean()); del X
fold, per, nrm = map(np.concatenate, (fold, per, nrm)); T = len(fold); gal = gallery(); P = np.unique(per); rng = np.random.default_rng(1)
B = mix11(per)  # 64-trial batches of 11 people picked at random (the sealed cohort has 11)
half = rng.permutation(P)[: len(P) // 2]  # one fixed random half of the people
SC = {'all at 0': {p: 0 for p in P}}
for dd in (-4, -2, 2, 4):
    SC['half at %+d, half at 0' % dd] = {p: (dd if p in half else 0) for p in P}
for lo, hi in ((-2, 2), (-4, 4), (-4, 0), (0, 4)):
    SC['spread %+d..%+d' % (lo, hi)] = dict(zip(P, rng.integers(lo, hi + 1, len(P))))


def batch_shifts(dl):  # per batch: (prominence, one correction c*) from the mixed curve; dl = per-trial delay
    out = []
    for b in B:
        ok = np.arange(-8, 9, 2); idx = np.clip(dl[b][:, None] + ok[None, :], -8, 8) + 8
        v = nrm[b][np.arange(len(b))[:, None], idx].mean(0); j = min(max(int(v.argmax()), 1), len(ok) - 2); l, m, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m + h
        out.append((v.max() / v.mean() - 1, int(round(min(max(ok[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.), ok[0]), ok[-1]) + 1.5))))
    return out


SH = {}
for name, dmap in SC.items():
    dl = np.array([dmap[p] for p in per]); est = batch_shifts(dl)
    SH[name, 'no delay'] = np.zeros(T, np.int8); SH[name, 'No alignment'] = np.clip(dl, -5, 5).astype(np.int8)
    for lab, g, z, cap, rej in POL:
        s = dl.copy()
        for b, (pr, c) in zip(B, est):
            if pr < g or abs(c) < z or (rej and abs(c) > cap):
                continue
            s[b] = dl[b] + max(-cap, min(cap, c))
        SH[name, lab] = np.clip(s, -5, 5).astype(np.int8)
H = {}
for k in KS:
    E = np.load(f'{OUT}/autolag_e2e_f{k}.npy'); rows = np.nonzero(fold == k)[0]; g4, tg = G4[k]; memo = {}
    for key, s in SH.items():
        sk = s[rows]; h = memo.get(sk.tobytes())
        if h is None:
            y = torch.as_tensor(E[sk.astype(int) + 5, np.arange(len(rows))].astype(np.float32)); h = memo[sk.tobytes()] = (rank(y[torch.as_tensor(g4[:, :3])].mean(1), tg, gal) < 5).numpy()
        H.setdefault(key, []).append(h)
    del E; print('fold', k, 'decoded', flush=True)
print('top-5 change against no alignment, pooled folds %s (± bootstrap); last column: the ceiling (same data, no delay) against no alignment' % KS)
for name in SC:
    b = np.concatenate(H[name, 'No alignment']).astype(float); bi = np.random.default_rng(0).integers(0, len(b), (500, len(b)))
    cell_ = lambda a: '%+6.1f ±%.1f' % (100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std())
    print('  %-24s | ' % name + ' | '.join('%s %s' % (lab.replace(' (finalA)', ''), cell_(np.concatenate(H[name, lab]).astype(float))) for lab, *_ in POL)
          + ' | ceiling %s' % cell_(np.concatenate(H[name, 'no delay']).astype(float)), flush=True)
