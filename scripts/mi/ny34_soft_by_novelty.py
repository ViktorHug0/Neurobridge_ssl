"""Y34 is the novelty effect of Y32 / Y33 the soft labels? Hard / soft twins ALONE, r=3 top-5 (four triplets), by novelty tercile of the
target (cosine to the nearest trained image): soft / hard ratio per tercile, folds pooled where both are cached.
    ny34_soft_by_novelty.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
TW = {'rmix': ('mi_rmix_ho5k_lag5_jit3:checkpoint_ep10.pth', 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth'),
      'ST-EEGFormer': ('mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth'),
      'LaBraM long': ('mi_plab_ho5k_lag5_long:checkpoint_ep10.pth', 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth'),
      'REVE-d6': ('mi_d6_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_d6_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth'),
      'REVE-scr': ('mi_scr_ho5k_lag5:checkpoint_test_best.pth', 'mi_scr_ho5k_lag5_soft3:checkpoint_test_best.pth'),
      'SQF-X': ('mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_sqfx_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth'),
      'SQF': ('mi_sqf_ho5k_lag5:checkpoint_ep32.pth', 'mi_sqf_ho5k_lag5_soft3:checkpoint_ep32.pth')}
gal = gallery(); bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)
NOV = torch.cat([(gal[i:i + 1000] @ bank.T).max(1).values.cpu() for i in range(0, len(gal), 1000)]).numpy(); Q3 = np.quantile(NOV, [1 / 3, 2 / 3]); H = {}
p = lambda k, rc: f"{OUT}/u01_f{k}_{rc.replace(':', '_')}.npy"
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    t = np.digitize(NOV[tg.numpy()], Q3)
    for name, (h, s) in TW.items():
        if os.path.exists(p(k, h)) and os.path.exists(p(k, s)):
            hh, ss = ((rank(scaled(torch.as_tensor(np.load(p(k, rc)).astype(np.float32)))[g3].mean(1), tg, gal) < 5).numpy().astype(float) for rc in (h, s))
            H.setdefault(name, []).append((hh, ss, t))
    print('fold', k, 'done', flush=True)
print('soft / hard twin alone, r=3 top-5: all | novel | middle | familiar (folds)')
for name, L in H.items():
    hh, ss, t = (np.concatenate([l[i] for l in L]) for i in range(3))
    print('  %-13s x%.3f | ' % (name, ss.mean() / hh.mean()) + ' | '.join('x%.3f' % (ss[t == j].mean() / hh[t == j].mean()) for j in range(3)) + ' (%d folds; hard %.3f %%)' % (len(L), 100 * hh.mean()), flush=True)
