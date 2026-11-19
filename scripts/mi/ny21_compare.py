"""Y21 package configurations against finalA's members (uniform weights, SQF-X ep40), no alignment in any (zero delay), 5k, r=3 (four
triplets), paired, per fold: D = finalD weights; E1 = finalD weights with SQF-X ep24; E2 = SQF-X ep24 with its own searched weights;
E2 + the ep40 SQF-X as a tenth member.
    ny21_compare.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
W1, W2 = [2, 1, 1.5, 1.5, 1, 1, 1, 1, 1.5], [1.5, .5, 1, 1, 1, .5, 1, 1, 1]
CFG = {'A finalA members, uniform': ('test_best', [1] * 9), 'D finalD weights': ('test_best', W1), 'E1 finalD weights, SQF-X ep24': ('ep24', W1),
       'E2 SQF-X ep24, own weights': ('ep24', W2)}
gal = gallery(big=bool(os.environ.get('BIG'))); H = {}  # BIG=1: the 14.4k THINGS gallery
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    Y = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{r}_{c}.npy").astype(np.float32)))
    P = [Y(*((r, c) if k != 2 or 'long_soft3' not in r else ('mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'))) for r, c in NINE]
    X24 = Y('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth')
    for lab, (e, w) in CFG.items():
        S = sum(wi * (X24 if (i == 0 and e == 'ep24') else p) for i, (wi, p) in enumerate(zip(w, P)))
        H.setdefault(lab, []).append((rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float))
        if lab.startswith('E2'):  # + the ep40 SQF-X as a tenth member (both SQF-X finals exist: no retraining)
            for x in (.5, 1.):
                H.setdefault('E2 + SQF-X ep40 x%.1f' % x, []).append((rank((S + x * P[0])[g3].mean(1), tg, gal) < 5).numpy().astype(float))
    del P, X24; print('fold', k, 'done', flush=True)
b = np.concatenate(H['A finalA members, uniform']); bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b)))
for lab, hs in H.items():
    a = np.concatenate(hs)
    print('%-32s top-5 %.4f %% | vs A %+.2f ± %.2f %% | per fold %s' % (lab, 100 * a.mean(), 100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(),
          ' '.join('%+.1f' % (100 * (x.mean() / y.mean() - 1)) for x, y in zip(hs, H['A finalA members, uniform']))), flush=True)
