"""Y35 a board-calibrated proxy: which novelty mix of the proxy's held-out images reproduces what the board measured for sub3 and sub4?
(Y34: soft labels are x0.6-0.8 on the third of images farthest from a trained image and x1.3-1.4 on the nearest third, x1.0 overall.)
Per (person, image) group: per-trial and r=3 hits at top-1 / top-5 for each configuration, and the target's novelty quintile. A mix is a
weight per quintile, w_q ~ exp(b * (q - 2)) (b < 0: more novel images than the proxy, b > 0: more familiar). For each b: the six board
statistics of sub3 -> sub4 (step ratios per-trial / aggregated x top-5 / top-1, aggregation gains of sub3 and sub4), chi-square against
the board, and what each candidate would be worth against sub4 under that mix.
    ny35_board_mix.py <folds>      (per-group table cached in y35_groups.npz; delete it to recompute)"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]; NPZ = f'{OUT}/y35_groups.npz'
SIX = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
W2 = [1.5, .5, 1, 1, 1, .5, 1, 1, 1]; SOFT = [4, 5, 6, 8]  # E2 weights; the soft-label slots of the nine
if not os.path.exists(NPZ):
    gal = gallery(); bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)
    NOV = torch.cat([(gal[i:i + 1000] @ bank.T).max(1).values.cpu() for i in range(0, len(gal), 1000)]).numpy(); G = {}
    for k in KS:
        X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
        for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
            groups.setdefault(kk, []).append(i)
        g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)]))
        tg = torch.as_tensor(img)[g3[:, 0]]; t1 = torch.as_tensor(img)[g4.ravel()]
        u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))
        P = [u(r, c) for r, c in NINE]; S = {'sub3': sum(u(r, c) for r, c in SIX), 'sub4': sum(P)}
        P[0] = u('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth')
        for lam in (0., .5, 1., 1.5, 2.):
            S['E2' if lam == 1 else 'E2 soft x%.1f' % lam] = sum(w * (lam if i in SOFT else 1.) * p for i, (w, p) in enumerate(zip(W2, P)))
        del P
        for name, Y in S.items():
            a = rank(Y[g4.ravel()], t1, gal).numpy().reshape(-1, 4); b = rank(Y[g3].mean(1), tg, gal).numpy().reshape(4, -1)
            G.setdefault(name, []).append(np.stack([(a < 5).mean(1), (a < 1).mean(1), (b < 5).mean(0), (b < 1).mean(0)], 1))  # per group: trial top-5, top-1, r=3 top-5, top-1
        G.setdefault('_nov', []).append(NOV[img[g4[:, 0]]]); G.setdefault('_person', []).append(subj[g4[:, 0]]); del S; print('fold', k, 'done', flush=True)
    np.savez(NPZ, **{n: np.concatenate(v) for n, v in G.items()})
Z = np.load(NPZ); nov = Z['_nov']; q = np.digitize(nov, np.quantile(nov, [.2, .4, .6, .8])); CFG = [n for n in Z.files if not n.startswith('_')]
T = {n: np.stack([Z[n][q == j].mean(0) for j in range(5)]) for n in CFG}  # (quintile, statistic) hit rates
print('novelty quintile edges (cosine to the nearest trained image):', np.round(np.quantile(nov, [.2, .4, .6, .8]), 3))
print('r=3 top-5 % by quintile (novel -> familiar): ' + ' | '.join('%s %s' % (n, ' '.join('%.2f' % (100 * v) for v in T[n][:, 2])) for n in ('sub3', 'sub4', 'E2', 'E2 soft x0.0')))
BOARD = [('step per-trial top-5', 1.083, .03), ('step aggregated top-5', 1.050, .03), ('step per-trial top-1', 1.064, .06), ('step aggregated top-1', 1.050, .06),
         ('gain sub3', 2.00, .05), ('gain sub4', 1.94, .05)]  # sub3 = sub16, sub4 = finalA; rough paired errors


def stats(w):
    m = {n: w @ T[n] for n in CFG}; a, b = m['sub3'], m['sub4']
    return [b[0] / a[0], b[2] / a[2], b[1] / a[1], b[3] / a[3], a[2] / a[0], b[2] / b[0]], m


print('mix w_q ~ exp(b (q - 2)); b < 0 = more novel than the proxy. Board: ' + ', '.join('%s %.3f' % (n, v) for n, v, _ in BOARD))
print('   b   | novel..familiar weights     | sub3->sub4: trial5 agg5 trial1 agg1 | gain3 gain4 | chi2 || vs sub4, r=3 top-5: E2 | soft x0 | x0.5 | x1.5 | x2')
for b in (-2., -1.5, -1., -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.5):
    w = np.exp(b * (np.arange(5) - 2)); w /= w.sum(); s, m = stats(w); chi = sum(((x - v) / e) ** 2 for x, (_, v, e) in zip(s, BOARD))
    print('%+5.2f | %s | %.3f %.3f %.3f %.3f | %.2f %.2f | %5.1f || %+5.1f %% | %+5.1f | %+5.1f | %+5.1f | %+5.1f' % (
        b, ' '.join('%.2f' % x for x in w), *s, chi, *(100 * (m[n][2] / m['sub4'][2] - 1) for n in ('E2', 'E2 soft x0.0', 'E2 soft x0.5', 'E2 soft x1.5', 'E2 soft x2.0'))), flush=True)
