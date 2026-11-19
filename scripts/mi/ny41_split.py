"""Y41 does a DIFFERENT conv-family run add when the family's weight is kept constant? (Y40: the two Squeezeformers are the weakest alone,
the least correlated with the token members (.60) and the most valuable (-6.1 % when dropped); every SQF-X variant read negative as a full
swap or as an addition, which changes either the incumbent or the family weight.) E2 with SQF-X's weight 1.5 split between the incumbent
(ep24) and a variant: 1.0 + 0.5 and 0.75 + 0.75; r=3 top-5, four triplets, paired against E2; folds with a cache only.
    ny41_split.py <folds> <recipe:checkpoint> ..."""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]; CAND = sys.argv[2:]
E2 = {'mi_sqfx_ho5k_lag5_jit3:checkpoint_ep24.pth': 1.5, 'mi_scr_ho5k_lag5:checkpoint_test_best.pth': .5, 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth': 1.,
      'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth': 1., 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth': .5,
      'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1., 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth': 1., 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth': 1.}
gal = gallery(); H = {}; f = lambda k, rc: f"{OUT}/u01_f{k}_{rc.replace(':', '_')}.npy"; u = lambda k, rc: scaled(torch.as_tensor(np.load(f(k, rc)).astype(np.float32)))
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    hit = lambda Y: (rank(Y[g3].mean(1), tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)
    S = sum(w * u(k, rc) for rc, w in E2.items()); inc = u(k, 'mi_sqfx_ho5k_lag5_jit3:checkpoint_ep24.pth'); H.setdefault('E2', {})[k] = hit(S)
    for c in CAND:
        if os.path.exists(f(k, c)):
            v = u(k, c)
            for a in (.5, .75):
                H.setdefault((c, a), {})[k] = hit(S + a * (v - inc))
    print('fold', k, 'done', flush=True)
for key, hs in H.items():
    if key != 'E2':
        ks = list(hs); a, b = np.concatenate([hs[k] for k in ks]), np.concatenate([H['E2'][k] for k in ks])
        print('%-58s %.2f of the 1.5: %+.2f ± %.2f %% | folds %s: %s' % (key[0].replace('checkpoint_', '').replace('_ho5k_lag5', ''), key[1], *paired(a, b, 1000), ks, ' '.join('%+.1f' % (100 * (hs[k].mean() / H['E2'][k].mean() - 1)) for k in ks)), flush=True)
