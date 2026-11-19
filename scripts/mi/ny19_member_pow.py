"""Y19 a norm power per MEMBER (Y11: one power on the ensemble norm is flat; soft and hard members may carry confidence differently).
Ensemble = sum_m w_m P_m |P_m|^(a_m - 1), P_m batch-scaled as deployed, weights = finalD's. Coordinate ascent over a_m in
{.5, .75, 1, 1.25, 1.5, 2} on the SELECT folds (5k, r=3, four triplets); the chosen powers against all a_m = 1 on the CHECK folds, paired.
    ny19_member_pow.py <select folds> <check folds>"""
from nc_lib import *

SEL, CHK = ([int(a) for a in s.split(',')] for s in sys.argv[1:3])
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 2.), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', 1.), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.5),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.5), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', 1.),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.5)]
NAMES = ['SQF-X', 'REVE-scr', 'SQF', 'LaBraM', 'rmix soft', 'LaBraM soft', 'ST soft', 'ST', 'REVE-pre soft']; gal = gallery(); D = {}
for k in SEL + CHK:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); T = np.unique(g4); loc = torch.as_tensor(np.searchsorted(T, g4))
    g3 = torch.cat([loc[:, [a for a in range(4) if a != o]] for o in range(4)])
    M = []
    for r, c, w in NINE:
        if k == 2 and 'long_soft3' in r:
            r, c = 'mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'
        P = w * scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{r}_{c}.npy').astype(np.float32)))[T]; M.append((P.half(), P.norm(dim=1)))
    D[k] = (M, g3, torch.as_tensor(img)[T][g3[:, 0]]); print('fold', k, 'loaded', flush=True)


def hits(k, a):
    M, g3, tg = D[k]; S = sum(P.float() * (nrm ** (e - 1))[:, None] for (P, nrm), e in zip(M, a))
    return (rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float)


score = lambda ks, a: np.mean(np.concatenate([hits(k, a) for k in ks]))
a = [1.] * 9; best = base = score(SEL, a); print('select %s: all a = 1 top-5 %.4f %%' % (SEL, 100 * base), flush=True)
for it in range(2):
    moved = False
    for j in range(9):
        for v in (.5, .75, 1., 1.25, 1.5, 2.):
            if v != a[j]:
                t = list(a); t[j] = v; s = score(SEL, t)
                if s > best + 1e-9:
                    best, a, moved = s, t, True
    print('round %d: %s, select %+.2f %%' % (it + 1, dict(zip(NAMES, a)), 100 * (best / base - 1)), flush=True)
    if not moved:
        break
for k in CHK:
    x, b = hits(k, a), hits(k, [1.] * 9); bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b)))
    print('check fold %d: %+.2f ± %.2f %%' % (k, 100 * (x.mean() / b.mean() - 1), 100 * (x[bi].mean(1) / b[bi].mean(1) - 1).std()), flush=True)
