"""Y39 two pure aggregation levers on the cached E2 outputs (the direction of a trial's output is untouched or nearly, so the per-trial score
cannot move: any change is a change of the gain from the grader's 3-repeat average).
 (a) artefact-aware trial weights from the INPUT (max |x|, rms over the epoch): a trial with a large artefact is down-weighted in the average
     (the output norm is the only weight today; Y36: 8.7 % of trials exceed 10 robust units, 2.3 % sit on the clamp at 20);
 (b) power-law compression / expansion of the output coordinates before the average (sign(y) |y|^p, norm kept), raw coordinates.
r=3 top-5, four triplets, paired against E2, per fold and pooled.
    ny39_aggw.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
gal = gallery(); H = {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; groups = {}
    mx, rms = X.abs().amax((1, 2)), X.pow(2).mean((1, 2)).sqrt(); del X
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
    u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))
    Y = sum(w * u(r, c) for r, c, w in E2); hit = lambda Q: (rank(Q[g3].mean(1), tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)
    med = torch.cat([rms[b:b + 64].median().expand(len(rms[b:b + 64])) for b in range(0, n, 64)])  # the batch's own median rms: stateless
    V = {'E2': Y}
    for c in (19.9, 10., 6.):
        for e in (0., .5):
            V['max|x| > %g: weight %.1f' % (c, e)] = Y * torch.where(mx > c, e, 1.)[:, None]
    for b in (.5, 1., 2.):
        V['(batch median rms / rms)^%.1f, capped at 1' % b] = Y * ((med / rms).clamp(max=1) ** b)[:, None]
    for b in (.5, 1.):
        V['(batch median rms / rms)^%.1f' % b] = Y * ((med / rms) ** b)[:, None]
    for p in (.75, .9, 1.1, 1.25):
        Z = Y.sign() * Y.abs() ** p; V['coordinates ^%.2f' % p] = Z * (Y.norm(dim=1) / Z.norm(dim=1))[:, None]
    for name, Q in V.items():
        H.setdefault(name, []).append(hit(Q))
    print('fold %d done; corr(log output norm, log rms) %.3f, (log norm, max|x| > 10) %.3f' % (k, np.corrcoef(Y.norm(dim=1).log().numpy(), rms.log().numpy())[0, 1], np.corrcoef(Y.norm(dim=1).log().numpy(), (mx > 10).float().numpy())[0, 1]), flush=True)
base = H['E2']; print('E2 r=3 top-5 %% per fold: ' + ' '.join('%.3f' % (100 * h.mean()) for h in base))
for name, hs in H.items():
    if name != 'E2':
        print('%-44s %+.2f ± %.2f %% | per fold %s' % (name, *paired(np.concatenate(hs), np.concatenate(base), 1000), ' '.join('%+.1f' % (100 * (h.mean() / b.mean() - 1)) for h, b in zip(hs, base))), flush=True)
