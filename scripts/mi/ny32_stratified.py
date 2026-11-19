"""Y32 where do the board steps come from? (19 Nov: the 5k judge predicted sub3 -> sub4 (x1.055 vs board x1.050) but not sub 2 -> sub3
(x1.10 vs x1.194). The sealed set is ~2x harder per trial and its ~40k images are mostly outside THINGS, while the proxy's held-out
images are new images of TRAINED concepts.) Same replay as ny31, r=3 top-5 hits per (person, image) group (four triplets), stratified by
  - person difficulty: the person's r=3 top-5 under sub3 (terciles of people), and
  - image novelty: cosine of the target to its nearest TRAINED image (11,540-image bank; terciles of images; low = novel).
Step ratios per stratum with a paired bootstrap; saved per-group hits: results/things_eeg/mi/nc/y32_hits.npz.
    ny32_stratified.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
SIX = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
W1, W2 = [2, 1, 1.5, 1.5, 1, 1, 1, 1, 1.5], [1.5, .5, 1, 1, 1, .5, 1, 1, 1]
gal = gallery(); bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)
NOV = torch.cat([(gal[i:i + 1000] @ bank.T).max(1).values.cpu() for i in range(0, len(gal), 1000)]).numpy()  # per held-out image: cosine to the nearest trained image
H, PER, IMG, FOLD = {}, [], [], []
for k in KS:
    X, subj, img = cell(FOLDS[k]); N = len(X); n = N // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))
    S = {}
    if k in ENS5:
        idx = np.argsort(np.random.default_rng(k * 100 + 7).permutation(N))[np.random.default_rng(7).permutation(N)][:n]
        S['sub 2'] = sum(scaled(torch.as_tensor(np.load(f'{R}/logs/holdout/f{k}/{m}.npy')[idx].astype(np.float32))) for m in ENS5[k])
    S['sub3'] = sum(u(r, c) for r, c in SIX); P = [u(r, c) for r, c in NINE]; S['sub4'] = sum(P); S['D'] = sum(w * p for w, p in zip(W1, P))
    P[0] = u('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth'); S['E2'] = sum(w * p for w, p in zip(W2, P)); del P
    for name, Y in S.items():  # hits per group = mean over its four triplets
        H.setdefault(name, {})[k] = (rank(Y[g3].mean(1), tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)
    PER.append(subj[g4[:, 0]]); IMG.append(img[g4[:, 0]]); FOLD.append(np.full(len(g4), k)); del S; print('fold', k, 'done', flush=True)
PER, IMG, FOLD = map(np.concatenate, (PER, IMG, FOLD)); nov = NOV[IMG]
full = lambda name: np.concatenate([H[name][k] if k in H[name] else np.full((FOLD == k).sum(), np.nan) for k in KS])
A = {name: full(name) for name in H}; np.savez(f'{OUT}/y32_hits.npz', person=PER, image=IMG, fold=FOLD, novelty=nov, **{n.replace(' ', ''): v for n, v in A.items()})
STEPS = [('sub 2', 'sub3', 'board x1.194'), ('sub3', 'sub4', 'board x1.050'), ('sub4', 'D', ''), ('sub4', 'E2', '')]


def ratio(a, b, m):  # pooled ratio b / a on the groups of mask m where both exist, paired bootstrap
    m = m & ~np.isnan(A[a]) & ~np.isnan(A[b]); x, y = A[a][m], A[b][m]; bi = np.random.default_rng(0).integers(0, len(x), (500, len(x)))
    return 'x%.3f ± %.3f' % (y.mean() / x.mean(), (y[bi].mean(1) / x[bi].mean(1)).std()) if m.sum() and x.mean() > 0 else '  -  '


P = np.unique(PER); base = np.array([A['sub3'][PER == p].mean() for p in P]); o = np.argsort(base)
print('per person (weakest first): sub3 r=3 top-5 %, then ' + ', '.join('%s->%s' % s[:2] for s in STEPS))
for i in o:
    print('  person %2d (fold %d) %.2f %% | ' % (P[i], FOLD[PER == P[i]][0], 100 * base[i]) + ' | '.join(ratio(a, b, PER == P[i]) for a, b, _ in STEPS), flush=True)
third = {p: t for t, ps in enumerate(np.array_split(P[o], 3)) for p in ps}; pt = np.array([third[p] for p in PER]); nt = np.digitize(nov, np.quantile(NOV, [1 / 3, 2 / 3]))
print('novelty terciles: nearest-trained-image cosine < %.3f | < %.3f | above; hit rate of sub3 by tercile: %s' % (*np.quantile(NOV, [1 / 3, 2 / 3]), ' '.join('%.2f %%' % (100 * A['sub3'][nt == t].mean()) for t in range(3))))
ALL = np.ones(len(PER), bool)
for lab, m in [('all', ALL)] + [('people: %s third' % n, pt == t) for t, n in enumerate(('weak', 'middle', 'strong'))] + [('images: %s' % n, nt == t) for t, n in enumerate(('novel third', 'middle third', 'familiar third'))] + \
        [('weak+middle people x novel images', (pt < 2) & (nt == 0)), ('strong people x familiar images', (pt == 2) & (nt == 2))]:
    print('%-34s (%5.1f %% of sub3 hits) | ' % (lab, 100 * A['sub3'][m].sum() / A['sub3'].sum()) + ' | '.join('%s->%s %s %s' % (a, b, ratio(a, b, m), bd) for a, b, bd in STEPS), flush=True)
