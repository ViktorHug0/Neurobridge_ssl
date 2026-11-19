"""Y37 is there aggregation headroom? (19 Nov board: our per-trial top-5 is 3rd, our gain from the 3-repeat average the second lowest.)
The grader averages the OUTPUTS of the repeats. Ceiling test with the frozen E2 fold members: decode the AVERAGE of the three input trials
(evidence added before the non-linear decoder) against the average of the three decoded outputs, r=3 top-5, four leave-one-out triplets.
If decoding the averaged input is much better, the members discard sub-threshold evidence per trial (they act as per-trial deciders).
'x1.73' = the averaged input rescaled by sqrt(3) (noise back at single-trial level, for the members without per-trial normalisation).
    ny37_coherent.py <fold>"""
from nc_lib import *

k = int(sys.argv[1])
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
t1 = torch.as_tensor(img)[g4.ravel()]; o = torch.randperm(len(g3), generator=torch.Generator().manual_seed(0))  # averaged inputs in mixed 64-batches, as single trials are
XA = torch.cat([X[g3[o[b:b + 4096]]].mean(1) for b in range(0, len(g3), 4096)])
inv = torch.argsort(o); hit = lambda q: (rank(q, tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)
SA, SB, SC = 0., 0., 0.; BS = int(os.environ.get('BS', 128))
print('member | per-trial top-5 %% | mean of 3 outputs (gain) | decode mean input: vs mean of outputs | same, input x1.73', flush=True)
for r, ck, w in E2:
    Y = scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{r}_{ck}.npy').astype(np.float32))); f = load(r, k, ck)
    a = hit(Y[g3].mean(1)); p1 = (rank(Y[g4.ravel()], t1, gal) < 5).float().mean().item()
    B = scaled(f(XA, bs=BS))[inv]; C = scaled(f(XA * 3 ** .5, bs=BS))[inv]; b, c = hit(B), hit(C); SA = SA + w * Y[g3].mean(1); SB = SB + w * B; SC = SC + w * C
    print('%-30s %.3f | %.3f (%.2f) | %.3f: %+.1f ± %.1f | %.3f: %+.1f ± %.1f' % (r[3:], 100 * p1, 100 * a.mean(), a.mean() / p1, 100 * b.mean(), *paired(b, a, 500), 100 * c.mean(), *paired(c, a, 500)), flush=True)
    del f; torch.cuda.empty_cache()
a, b, c = hit(SA), hit(SB), hit(SC)
print('E2: mean of outputs %.3f | decode mean input %.3f: %+.1f ± %.1f | x1.73 %.3f: %+.1f ± %.1f' % (100 * a.mean(), 100 * b.mean(), *paired(b, a, 1000), 100 * c.mean(), *paired(c, a, 1000)), flush=True)
