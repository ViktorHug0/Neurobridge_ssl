"""Y29 move each trial's output along its neighbourhood in the TRAINING-image bank (11,540 DINOv2 embeddings, never the test images):
q' = |q| * nz(nz(q) + a * nz(sum_j softmax(nz(q) . b_j / tau) b_j)). a > 0 pulls the output onto the manifold of plausible images
(a test-time "soft label"); a < 0 pushes it away from dense regions (query-side anti-hubness). Nonlinear and per trial, unlike the
linear output maps closed on 10 Nov. Stateless (the bank is a constant). E2, 5k, r=3 (four triplets), paired against E2.
    ny29_bank.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)
GRID = [(a, t) for a in (-.3, -.15, -.05, .05, .15, .3) for t in (.02, .05)]
gal = gallery(); H = {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); T = np.unique(g4); loc = torch.as_tensor(np.searchsorted(T, g4))
    g3 = torch.cat([loc[:, [a for a in range(4) if a != o]] for o in range(4)]); tg = torch.as_tensor(img)[T][g3[:, 0]]
    S = 0
    for r, c, w in E2:
        if k == 2 and 'long_soft3' in r:
            r, c = 'mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'
        S = S + w * scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{r}_{c}.npy').astype(np.float32)))[T]
    H.setdefault((0, 0), []).append((rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float))
    nrm, d = S.norm(dim=1, keepdim=True), nz(S)
    for tau in (.02, .05):
        m = torch.cat([nz(torch.softmax(d[i:i + 4096].to(DEV) @ bank.T / tau, 1) @ bank).cpu() for i in range(0, len(d), 4096)])
        for a in (-.3, -.15, -.05, .05, .15, .3):
            Q = nrm * nz(d + a * m)
            H.setdefault((a, tau), []).append((rank(Q[g3].mean(1), tg, gal) < 5).numpy().astype(float))
    del S, d, m; print('fold', k, 'done', flush=True)
b = H[0, 0]
for key in GRID:
    x, y = np.concatenate(H[key]), np.concatenate(b); bi = np.random.default_rng(0).integers(0, len(y), (500, len(y)))
    print('a %+.2f tau %.2f: %+.2f ± %.2f %% | per fold %s' % (*key, 100 * (x.mean() / y.mean() - 1), 100 * (x[bi].mean(1) / y[bi].mean(1) - 1).std(),
          ' '.join('%+.1f' % (100 * (p.mean() / q.mean() - 1)) for p, q in zip(H[key], b))), flush=True)
