"""Y09 per-trial weighting of the ensemble output before the grader's repeat average: y -> y |y|^(a-1), paired against a = 1 (as
deployed). sub25b's nine (fold models, cached u01 outputs), held-out people x 5,000 never-trained images, all four leave-one-out
triplets of the 4 repeats, 5k and 14k galleries; paired bootstrap over (person, image).
    ny09_pow_paired.py <folds> [powers, default 1.25,1.5,1.75,2]"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]; PW = [float(a) for a in (sys.argv[2] if len(sys.argv) > 2 else '1.25,1.5,1.75,2').split(',')]
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
gals = {'5k': gallery(), '14k': gallery(big=True)}; H = {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; groups = {}; del X
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)]); tg = torch.as_tensor(img)[g3[:, 0]]
    rk = lambda r, c: f"{OUT}/u01_f{k}_{r if glob.glob(f'{ROOT}/{r}/fold{k}/*/*/{c}') else r.replace('_jit3', '')}_{c}.npy"
    S = sum(scaled(torch.as_tensor(np.load(rk(r, c)).astype(np.float32))) for r, c in NINE); nrm = S.norm(dim=1, keepdim=True)
    for a in [1.0] + PW:
        y = S * nrm ** (a - 1)
        for gn, g in gals.items():
            h = (rank(y[g3].mean(1), tg, g) < 5).numpy().astype(float); H.setdefault((a, gn), []).append(h.reshape(4, -1).mean(0))
    print('fold', k, 'done', flush=True)
for gn in gals:
    b = np.concatenate(H[(1.0, gn)]); bi = np.random.default_rng(0).integers(0, len(b), (2000, len(b)))
    for a in PW:
        x = np.concatenate(H[(a, gn)]); per = ' / '.join('%+.1f' % (100 * (u.mean() / v.mean() - 1)) for u, v in zip(H[(a, gn)], H[(1.0, gn)]))
        print('%-4s pow %.2f vs 1: top-5 %+.2f ± %.2f %% | per fold %s' % (gn, a, 100 * (x.mean() / b.mean() - 1), 100 * (x[bi].mean(1) / b[bi].mean(1) - 1).std(), per), flush=True)
