"""Y31 replay of the scored submissions on the 5k judge (user, 19 Nov: how well did 5k predict each board step?). Fold-model stand-ins,
scaled sums, no alignment (zero delay), held-out people x 5k never-trained images. sub 2 = ENS5 (d6, rmix, SQF-X, REVE-scratch, SQF; the
pre-lag cv4 recipes, old cache logs/holdout/f<k> re-ordered from its seed k*100+7 shuffle, folds 1-3 only); sub3 = sub16's six (lag + jitter + LaBraM); sub4 = finalA's nine
(uniform); E2 = finalE. Per-trial top-5 (r=1), r=3 top-5 (four leave-one-out triplets), aggregation gain, and step ratios per fold,
against the board: sub 2 -> sub3 x1.135 per-trial / x1.194 aggregated; sub3 -> sub4 x1.083 / x1.050.
    ny31_board_replay.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
SIX = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth', 1), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1),
       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', 1), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth', 1)]
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
W2 = [1.5, .5, 1, 1, 1, .5, 1, 1, 1]
gal = gallery(); RES = {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); N = len(X); n = N // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)]))
    tg = torch.as_tensor(img)[g3[:, 0]]; t1 = torch.as_tensor(img)[g4.ravel()]
    u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))
    S = {}
    if k in ENS5:  # the old cache: rows in cell() order, 100,000 trials -> first n
        idx = np.argsort(np.random.default_rng(k * 100 + 7).permutation(N))[np.random.default_rng(7).permutation(N)][:n]  # old cache: logs/holdout_eval.py shuffled with seed k*100+7
        S['sub 2'] = sum(scaled(torch.as_tensor(np.load(f'{R}/logs/holdout/f{k}/{m}.npy')[idx].astype(np.float32))) for m in ENS5[k])
    S['sub3'] = sum(u(r, c) for r, c, _ in SIX)
    P = [u(r, c) for r, c in NINE]; S['sub4'] = sum(P)
    P[0] = u('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth'); S['E2'] = sum(w * p for w, p in zip(W2, P)); del P
    for name, Y in S.items():
        RES[name, k] = ((rank(Y[g4.ravel()], t1, gal) < 5).numpy().astype(float), (rank(Y[g3].mean(1), tg, gal) < 5).numpy().astype(float))
    del S; print('fold', k, 'done', flush=True)
print('per fold: per-trial top-5 %% | r=3 top-5 %% | gain')
for k in KS:
    print('  fold %d: ' % k + ' | '.join('%s %.3f / %.3f / %.2f' % (nm, 100 * RES[nm, k][0].mean(), 100 * RES[nm, k][1].mean(), RES[nm, k][1].mean() / RES[nm, k][0].mean())
                                       for nm in ('sub 2', 'sub3', 'sub4', 'E2') if (nm, k) in RES), flush=True)
for a, b, board in (('sub 2', 'sub3', (1.135, 1.194)), ('sub3', 'sub4', (1.083, 1.050)), ('sub4', 'E2', None)):
    ks = [k for k in KS if (a, k) in RES]
    if not ks:
        continue
    rr = [np.mean(np.concatenate([RES[b, k][i] for k in ks])) / np.mean(np.concatenate([RES[a, k][i] for k in ks])) for i in (0, 1)]
    pf = ' '.join('%.3f/%.3f' % (RES[b, k][0].mean() / RES[a, k][0].mean(), RES[b, k][1].mean() / RES[a, k][1].mean()) for k in ks)
    print('%s -> %s, folds %s: per-trial x%.3f, aggregated x%.3f%s | per fold (per-trial/agg) %s' % (a, b, ks, *rr, ' | board x%.3f / x%.3f' % board if board else '', pf), flush=True)
