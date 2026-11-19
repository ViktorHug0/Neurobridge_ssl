"""Y36 amplitude outliers at r=3 (audit, 19 Nov: the grader chain clamps at +-20 robust units and no tighter clamp was ever tried; at ~3 repeats
one artefact trial is not averaged away; NEAR reports per-trial wavelet thresholding as the only denoiser that helps on Alljoined-1.6M).
Zero-shot: the frozen E2 fold members on inputs clipped at +-c, r=3 top-5 (four leave-one-out triplets) against c = 20, per member and for E2.
    ny36_clip.py <fold> [c ...]"""
from nc_lib import *

k = int(sys.argv[1]); CS = [float(a) for a in sys.argv[2:]] or [10., 6., 4., 3.]
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
hit = lambda Y: (rank(Y[g3].mean(1), tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)
a = X.abs(); print('fold %d, %d trials. Share of samples / of trials with any sample beyond c: ' % (k, n) + ' | '.join('c=%g: %.3f %% / %.1f %%' % (c, 100 * (a > c).float().mean(), 100 * (a.amax((1, 2)) > c).float().mean()) for c in [19.9] + CS), flush=True)
print('  per person, trials beyond c=%g: ' % CS[-2] + ' '.join('%d: %.0f %%' % (s, 100 * (a[torch.as_tensor(subj == s)].amax((1, 2)) > CS[-2]).float().mean()) for s in FOLDS[k])); del a
S = {c: 0. for c in [20.] + CS}
for r, ck, w in E2:
    f = load(r, k, ck); base = None
    for c in S:
        y = scaled(f(X if c == 20. else X.clamp(-c, c), bs=int(os.environ.get('BS', 256)))); S[c] = S[c] + w * y; h = hit(y)
        if c == 20.:
            base = h; print('%-32s c=20: %.3f %%' % (r[3:], 100 * h.mean()), end='')
        else:
            print(' | c=%g: %+.1f ± %.1f' % (c, *paired(h, base, 500)), end='')
    print(flush=True); del f; torch.cuda.empty_cache()
base = hit(S[20.]); print('E2 c=20: %.3f %%' % (100 * base.mean()) + ''.join(' | c=%g: %+.2f ± %.2f' % (c, *paired(hit(S[c]), base, 1000)) for c in CS), flush=True)
