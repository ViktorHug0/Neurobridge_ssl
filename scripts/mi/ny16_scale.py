"""Y16 member scaling: per 64-batch mean norm (deployed, `scaled`) vs one constant per member (its mean norm over the fold; in a
package a constant measured on training data). The batch factor changes a trial's weight in the grader's repeat average with the
batch's other people. Weighted nine, 5k, r=3 (four triplets), paired, per fold.
    ny16_scale.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
NINE = {'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth': 2., 'mi_scr_ho5k_lag5:checkpoint_test_best.pth': 1., 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth': 1.,
        'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth': 1.5, 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth': 1., 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth': 1.,
        'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1., 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1.}
gal = gallery(); H = {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    B, G = 0, 0
    for rc, w in NINE.items():
        if k == 2 and 'long_soft3' in rc:
            rc = 'mi_plab_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth'
        P = torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{rc.replace(':', '_')}.npy").astype(np.float32)); B = B + w * scaled(P); G = G + w * P / P.norm(dim=1).mean(); del P
    for lab, S in (('batch', B), ('constant', G)):
        H.setdefault(lab, []).append((rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float))
    print('fold', k, 'done', flush=True)
a, b = np.concatenate(H['constant']), np.concatenate(H['batch']); bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b)))
print('constant vs batch scaling: %+.2f ± %.2f %% | per fold %s' % (100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(),
      ' '.join('%+.1f' % (100 * (x.mean() / y.mean() - 1)) for x, y in zip(H['constant'], H['batch']))))
