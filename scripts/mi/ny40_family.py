"""Y40 how the E2 members differ (user, 19 Nov: which inductive family is under-represented?). Per member on the given folds: r=3 top-5 alone,
gain, and the correlation between members of the target's z-score among the gallery (per r=3 query, first triplet): high = same errors.
    ny40_family.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
E2 = [('SQF-X', 'mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth'), ('SQF', 'mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('REVE-scr', 'mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'),
      ('rmix soft', 'mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('LaBraM', 'mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('LaBraM soft', 'mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
      ('ST', 'mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('ST soft', 'mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('REVE-pre soft', 'mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
gal = gallery(); Z = {n: [] for n, _, _ in E2}; H = {n: [] for n, _, _ in E2}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
    for name, r, c in E2:
        if k == 2 and 'long_soft3' in r:
            r, c = 'mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'
        q = nz(scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{r}_{c}.npy').astype(np.float32)))[g4[:, :3]].mean(1)); z, h = [], []
        for b in range(0, len(q), 4096):
            S = q[b:b + 4096].to(DEV) @ gal.T; t = S.gather(1, tg[b:b + 4096].to(DEV)[:, None]); z.append(((t - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)).squeeze(1).cpu()); h.append(((S > t).sum(1) < 5).cpu())
        Z[name].append(torch.cat(z)); H[name].append(torch.cat(h).float())
names = [n for n, _, _ in E2]; Zm = torch.stack([torch.cat(Z[n]) for n in names]); Cm = torch.corrcoef(Zm)
print('r=3 top-5 %% alone: ' + ' | '.join('%s %.2f' % (n, 100 * torch.cat(H[n]).mean()) for n in names))
print('correlation of the target z-score (rows / columns in the same order):'); print(' ' * 14 + ' '.join('%8s' % n[:8] for n in names))
for i, n in enumerate(names):
    print('%-14s' % n + ' '.join('%8.2f' % Cm[i, j] for j in range(len(names))))
conv, tok = [0, 1], [2, 3, 4, 5, 6, 7, 8]; m = lambda A, B: np.mean([Cm[i, j].item() for i in A for j in B if i != j])
print('mean correlation: conv-conv %.2f | token-token %.2f | conv-token %.2f | scratch tokens (REVE-scr, rmix) with pretrained %.2f | among pretrained %.2f' % (m(conv, conv), m(tok, tok), m(conv, tok), m([2, 3], [4, 5, 6, 7, 8]), m([4, 5, 6, 7, 8], [4, 5, 6, 7, 8])))
