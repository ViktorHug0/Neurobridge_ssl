"""S09 which corrections should the alignment apply? (user, 18 Nov: price on four folds, people that wreck it, per-lag reliability, dead zone)
Per-trial detector norms (lag-only rmix of each fold at shifts -8..+8, `s06_trialnorm_f<k>.npy`) and cached decoding at total shifts
-5..+5 (`autolag_e2e_f<k>.npy`); held-out people x 5,000 never-trained images, r=3, 5k gallery, top-5 against no alignment.
Batches: 'sealed-like' = 64 trials from 11 people picked at random among ALL given folds (the sealed cohort has 11); 'one fold' = 64 trials of one fold's five people;
'one person' = 64 trials of one person. True delay d = -4..+4 (data received d samples late). Estimator = the package's (parabola
vertex + 1.5, rounded); policies: gate (skip flat curves) x dead zone (skip corrections smaller than D samples).
Prints (1) P(proposed correction | true delay), (2) top-5 change per policy and delay, (3) per person, one-person batches.
    ns09_align_policy.py <folds, e.g. 1,2,3,4>"""
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

KS = [int(a) for a in sys.argv[1].split(',')]; C = np.arange(-8, 9); D = list(range(-4, 5))
# policies: (label, gate, dead zone, cap, reject). A proposal is skipped below the gate or inside the dead zone; above the cap it is clipped
# to the cap, or (reject) skipped. Every policy is also bounded by the cached outputs: total shifts outside -5..+5 read as +-5.
POL = [('Always', 0., 0, 99, False), ('Gate 0.10 (finalA)', .1, 0, 99, False), ('Dead zone 2', 0., 2, 99, False), ('Dead zone 3', 0., 3, 99, False),
       ('Gate + dead zone 2', .1, 2, 99, False), ('Gate + dead zone 3', .1, 3, 99, False), ('Gate + cap 5', .1, 0, 5, False),
       ('Gate + reject beyond 4', .1, 0, 4, True), ('Gate + reject beyond 3', .1, 0, 3, True), ('Gate + clip at 3', .1, 0, 3, False)]
fold, loc, per, nrm, G4 = [], [], [], [], {}
for k in KS:
    p = f'{OUT}/s06_trialnorm_f{k}.npy'; X, subj, img = cell(FOLDS[k]); n = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r').shape[1]
    if not os.path.exists(p):  # as ns06: per-trial norm of the fold's lag-only rmix at every candidate shift
        f = load('mi_rmix_ho5k_lag5', k, 'checkpoint_ep10.pth'); tt = torch.arange(120); Xn = torch.as_tensor(X[:n])
        np.save(p, np.stack([f(Xn[:, :, (tt - int(c)).clamp(0, 119)], bs=1000).norm(dim=1).numpy() for c in C], 1)); del f, Xn
    N = np.load(p); subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); G4[k] = (g4, torch.as_tensor(img)[g4[:, 0]], subj[g4[:, 0]])
    fold.append(np.full(n, k)); loc.append(np.arange(n)); per.append(subj); nrm.append(N / N.mean()); del X
fold, loc, per, nrm = map(np.concatenate, (fold, loc, per, nrm)); T = len(fold); gal = gallery()
order = np.random.default_rng(0).permutation(T)
B = {'sealed-like': mix11(per),  # 11 people per batch, as in the sealed cohort
     'one fold': [np.nonzero(fold == k)[0][i:i + 64] for k in KS for i in range(0, (fold == k).sum() - 31, 64)],
     'one person': [q[i:i + 64] for q in (np.nonzero(per == s)[0] for s in np.unique(per)) for i in range(0, len(q) - 31, 64)]}


def estimate(b, d):  # the package's estimator on batch b received with delay d: (curve prominence, total shift after correction)
    ok = (np.abs(C - d) <= 8) & ((C - d) % 2 == 0); c = C[ok]; v = nrm[b][:, ok].mean(0); j = min(max(int(v.argmax()), 1), len(c) - 2)
    l, m, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m + h
    return v.max() / v.mean() - 1, int(round(min(max(c[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.), c[0]), c[-1]) + 1.5))


EST = {(name, d): [estimate(b, d) for b in bs] for name, bs in B.items() for d in D}
print('(1) proposed correction (total shift - delay) by true delay, % of batches; ideal = -d or 1 - d (decoding optimum +0.7)')
for name in ('sealed-like', 'one person'):
    print('  %s: corrections -6..+6' % name)
    for d in D:
        cor = np.array([t - d for _, t in EST[name, d]]); print('    d %+d: ' % d + ' '.join('%4.0f' % (100 * (cor == c).mean()) for c in range(-6, 7)))
SH = {}  # per-trial total shift for every (batch type, delay, policy), and the baseline (no alignment: shift = delay)
for name, bs in B.items():
    for d in D:
        SH[name, d, None] = np.full(T, d, np.int8)
        for lab, g, z, cap, rej in POL:
            s = np.full(T, d, np.int8)
            for b, (pr, t) in zip(bs, EST[name, d]):
                c = t - d
                if pr < g or abs(c) < z or (rej and abs(c) > cap):
                    continue
                s[b] = d + max(-cap, min(cap, c))
            SH[name, d, lab] = np.clip(s, -5, 5)
H = {}  # hits per key, concatenated over folds in KS order; per-fold lists kept for the per-fold and per-person reads
for k in KS:
    E = np.load(f'{OUT}/autolag_e2e_f{k}.npy'); rows = np.nonzero(fold == k)[0]; g4, tg, sp = G4[k]; memo = {}
    for key, s in SH.items():
        sk = s[rows]; h = memo.get(sk.tobytes())
        if h is None:
            y = torch.as_tensor(E[sk.astype(int) + 5, np.arange(len(rows))].astype(np.float32))
            h = memo[sk.tobytes()] = (rank(y[torch.as_tensor(g4[:, :3])].mean(1), tg, gal) < 5).numpy()
        H.setdefault(key, []).append(h)
    del E; print('fold', k, 'decoded:', len(memo), 'distinct shift patterns', flush=True)
PER = np.concatenate([G4[k][2] for k in KS])
def ch(a, b, bi=None):
    v = 100 * (a.mean() / b.mean() - 1); return '%+5.1f' % v if bi is None else '%+5.1f ±%.1f' % (v, 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std())
print('(2) top-5 change against no alignment, pooled over folds %s (± bootstrap); at d = 0 also per fold' % KS)
for name in B:
    print('  %s' % name)
    for lab, g, z, cap, rej in POL:
        cells = []
        for d in D:
            a, b = np.concatenate(H[name, d, lab]), np.concatenate(H[name, d, None]); bi = np.random.default_rng(0).integers(0, len(a), (500, len(a)))
            cells.append(('d%+d %s' % (d, ch(a, b, bi))) if d == 0 else 'd%+d %s' % (d, ch(a, b)))
        f0 = ' / '.join(ch(x, y) for x, y in zip(H[name, 0, lab], H[name, 0, None]))
        print('    %-24s | %s | d0 per fold %s' % (lab, ' | '.join(cells), f0), flush=True)
print('(3) one-person batches, per person: top-5 change against no alignment | mean applied correction (d = 0, then d = -4)')
for lab, g, z, cap, rej in POL:
    for d in (0, -4):
        a, b, s = np.concatenate(H['one person', d, lab]), np.concatenate(H['one person', d, None]), SH['one person', d, lab]
        print('    %-24s d %+d: ' % (lab, d) + ' '.join('p%d %s (%+.1f)' % (q, ch(a[PER == q], b[PER == q]), (s[per == q] - d).mean()) for q in np.unique(PER)), flush=True)
# per-person data for the plots (scripts/mi/ns09_plot.py): one-person batches (and, for comparison, the same people inside sealed-like
# batches), d = 0 and d = -4: top-5 change against no alignment per person with a bootstrap sd; every one-person batch's proposed correction
rng = np.random.default_rng(0); P = np.unique(PER); out = {'persons': P, 'fold_of_person': np.array([fold[per == q][0] for q in P])}
for name in ('one person', 'sealed-like'):
    for d in (0, -4):
        for lab, g, z, cap, rej in POL:
            a, b = np.concatenate(H[name, d, lab]).astype(float), np.concatenate(H[name, d, None]).astype(float); gs = []
            for q in P:
                m = np.nonzero(PER == q)[0]; bi = m[rng.integers(0, len(m), (500, len(m)))]
                gs.append((100 * (a[m].mean() / b[m].mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(), 100 * b[m].mean()))
            out['%s|d%d|%s' % (name, d, lab)] = np.array(gs)
for d in (0, -4):
    out['batch_person_d%d' % d] = np.array([per[b[0]] for b in B['one person']]); out['batch_corr_d%d' % d] = np.array([t - d for _, t in EST['one person', d]])
    out['batch_prom_d%d' % d] = np.array([pr for pr, _ in EST['one person', d]])
np.savez(f"{OUT}/s09_person_f{''.join(map(str, KS))}.npz", **out); print('saved', f"{OUT}/s09_person_f{''.join(map(str, KS))}.npz")
