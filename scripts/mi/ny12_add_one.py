"""Y12 does any cached fold model add to the weighted nine (E2 since 19 Nov 03:45: SQF-X ep24 1.5, REVE-scr .5, LaBraM soft .5, others 1) on the 5k proxy? One candidate at a time,
weight 0.5 / 1, r=3 top-5 (four leave-one-out triplets), paired against the nine; per fold and pooled. Folds with no cache are skipped.
'rc@avg' = the member's mean output over input shifts -1, 0, +1 (u01s+-1 caches). 'old>new' swaps a member of the nine for another at the same weight ('old>' drops it); 'a>b+c>d' does several swaps at once.
    ny12_add_one.py <folds> <recipe:checkpoint | old>new ...> ..."""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]; CAND = sys.argv[2:]
NINE = {'mi_sqfx_ho5k_lag5_jit3:checkpoint_ep24.pth': 1.5, 'mi_scr_ho5k_lag5:checkpoint_test_best.pth': .5, 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth': 1.,  # E2 (Y21)
        'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth': 1., 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth': .5,
        'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1., 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1.}
gal = gallery(); H = {}
f = lambda k, rc: f"{OUT}/u01_f{k}_{rc.replace(':', '_')}.npy"
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)]))
    tg = torch.as_tensor(img)[g3[:, 0]]; Y1 = lambda p: scaled(torch.as_tensor(np.load(p).astype(np.float32)))
    Y = lambda rc: Y1(f(k, rc)) if not rc.endswith('@avg') else sum(Y1(f(k, rc[:-4]).replace('/u01_', '/u01s%d_' % d if d else '/u01_')) for d in (-1, 0, 1)) / 3  # '@avg': mean over input shifts -1, 0, +1
    S = sum(w * Y(rc if k != 2 or 'long_soft3' not in rc else 'mi_plab_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth') for rc, w in NINE.items())
    hit = lambda S: (rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float)
    H[k, None] = hit(S)
    for rc in [c for c in CAND if '>' in c]:
        if not all(os.path.exists(f(k, sw.split('>')[1].removesuffix('@avg'))) for sw in rc.split('+') if sw.split('>')[1]):
            continue  # a new member without a cache on this fold
        T = S.clone()
        for sw in rc.split('+'):
            old, new = sw.split('>'); T -= NINE[old] * Y(old if k != 2 or 'long_soft3' not in old else 'mi_plab_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth')
            if new:
                T += NINE[old] * Y(new)
        H[k, rc, .5] = H[k, rc, 1.] = hit(T); del T
    for rc in [c for c in CAND if '>' not in c]:
        if os.path.exists(f(k, rc)):
            P = Y(rc)
            for w in (.5, 1.):
                H[k, rc, w] = hit(S + w * P)
            del P
    print('fold', k, 'done', flush=True)
for rc in CAND:
    for w in (.5, 1.):
        ks = [k for k in KS if (k, rc, w) in H]
        if ks:
            a, b = np.concatenate([H[k, rc, w] for k in ks]), np.concatenate([H[k, None] for k in ks]); bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b)))
            print('%-60s w %.1f: %+.2f ± %.2f %% | folds %s: %s' % (rc, w, 100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(), ks,
                  ' '.join('%+.1f' % (100 * (H[k, rc, w].mean() / H[k, None].mean() - 1)) for k in ks)), flush=True)
