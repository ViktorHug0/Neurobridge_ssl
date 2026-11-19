"""Y13 spectral sharpening of the output (Y11: soft labels raise per-trial top-5 but lower the gain of the repeat average, a bias the
average cannot remove). Fixed map q -> V diag(lam^-g) V^T q, V / lam = eigen-axes of the training-image embeddings (11,540 images,
second moment 'u' or centred covariance 'c'); no state, one 1536x1536 matrix in the package. Weighted nine (SQF-X 2, LaBraM 1.5,
others 1) and single members; held-out people x 5k never-trained images, 5k gallery, per-trial and r=3 top-5 (four triplets), paired.
    ny13_spectral.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
NINE = {'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth': 2., 'mi_scr_ho5k_lag5:checkpoint_test_best.pth': 1., 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth': 1.,
        'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth': 1.5, 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth': 1., 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth': 1.,
        'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1., 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1.}
SOLO = {'rmix soft': 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth', 'rmix': 'mi_rmix_ho5k_lag5_jit3:checkpoint_ep10.pth',
        'ST soft': 'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth', 'SQF-X': 'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth'}
T = torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).double(); T = T / T.norm(dim=1, keepdim=True)
E = {'u': torch.linalg.eigh(T.T @ T / len(T)), 'c': torch.linalg.eigh(torch.cov(T.T))}
G = [0., .1, .25, .5, .75]; MAP = {(b, g): (E[b][1] * E[b][0].clamp_min(1e-8) ** -g) @ E[b][1].T for b in E for g in G if g}
gal = gallery(); H = {}
f = lambda k, rc: f"{OUT}/u01_f{k}_{(rc if k != 2 or 'long_soft3' not in rc else 'mi_plab_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth').replace(':', '_')}.npy"
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)]))
    tg = torch.as_tensor(img)[g3[:, 0]]; t1 = torch.as_tensor(img)[g4.ravel()]; Y = lambda rc: scaled(torch.as_tensor(np.load(f(k, rc)).astype(np.float32)))
    out = {'nine': sum(w * Y(rc) for rc, w in NINE.items())}; out.update({m: Y(rc) for m, rc in SOLO.items()})
    for m, S in out.items():
        for key in [('u', 0.)] + list(MAP):
            Q = S if not key[1] else (S.double() @ MAP[key]).float()
            H.setdefault((m, key), []).append(((rank(Q[g4.ravel()], t1, gal) < 5).numpy().astype(float), (rank(Q[g3].mean(1), tg, gal) < 5).numpy().astype(float)))
    del out; print('fold', k, 'done', flush=True)
print('per-trial top-5 / r=3 top-5 change against the raw output (g = 0), pooled folds %s; per fold r=3' % KS)
for (m, key), hs in H.items():
    if key[1]:
        b = H[m, ('u', 0.)]; r1 = [np.concatenate([h[i] for h in hs]) for i in (0, 1)]; r0 = [np.concatenate([h[i] for h in b]) for i in (0, 1)]
        bi = np.random.default_rng(0).integers(0, len(r0[1]), (500, len(r0[1])))
        print('  %-10s %s g %.2f: per-trial %+5.1f %% | r=3 %+5.1f ± %.1f %% | per fold %s' % (m, *key, 100 * (r1[0].mean() / r0[0].mean() - 1), 100 * (r1[1].mean() / r0[1].mean() - 1),
              100 * (r1[1][bi].mean(1) / r0[1][bi].mean(1) - 1).std(), ' '.join('%+.1f' % (100 * (x[1].mean() / y[1].mean() - 1)) for x, y in zip(hs, b))), flush=True)
