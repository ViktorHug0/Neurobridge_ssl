"""Y33 soft-label members by image novelty (Y32: sub3 -> sub4 is x0.83 on the third of held-out images farthest from any trained image and
x1.25 on the nearest third). E2 split into H = its five hard-label members and F = its four soft-label members (E2 weights inside each).
(1) oracle: r=3 top-5 of H + lam * F per novelty tercile of the TARGET (cosine to the nearest trained image): how far apart are the optima?
(2) can a trial tell? per-trial features against the target's novelty (Spearman): bank peak of the E2 output, of H, of F (max cosine to
    the 11,540 trained images), cosine(H, F), |F| / |H|;
(3) routing: lam = lo below the feature's median, hi above (median from the SELECT folds); grid chosen on SELECT, read on CHECK, vs E2.
    ny33_novelty_route.py <select folds> <check folds>"""
from nc_lib import *
from scipy.stats import spearmanr

SEL, CHK = ([int(a) for a in s.split(',')] for s in sys.argv[1:3])
HARD = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.)]
SOFT = [('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
gal = gallery(); bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)
NOV = torch.cat([(gal[i:i + 1000] @ bank.T).max(1).values.cpu() for i in range(0, len(gal), 1000)]).numpy(); Q3 = np.quantile(NOV, [1 / 3, 2 / 3])
peak = lambda S: torch.cat([(nz(S[i:i + 4096]).to(DEV) @ bank.T).max(1).values.cpu() for i in range(0, len(S), 4096)])
LAM = (0., .5, 1., 1.5, 2., 3.); D = {}
for k in SEL + CHK:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); T = np.unique(g4); loc = torch.as_tensor(np.searchsorted(T, g4))
    g3 = torch.cat([loc[:, [a for a in range(4) if a != o]] for o in range(4)]); tg = torch.as_tensor(img)[T][g3[:, 0]]
    u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))[T]
    H = sum(w * u(r, c) for r, c, w in HARD); F = sum(w * u(r, c) for r, c, w in SOFT)
    feats = {'bank peak E2': peak(H + F), 'bank peak H': peak(H), 'bank peak F': peak(F), 'cos(H, F)': (nz(H) * nz(F)).sum(1), '|F| / |H|': F.norm(dim=1) / H.norm(dim=1)}
    D[k] = (H, F, g3, tg, feats, torch.as_tensor(NOV[img[T]])); print('fold', k, 'loaded', flush=True)


def hits(k, lam):  # lam: scalar or per-trial tensor
    H, F, g3, tg = D[k][:4]; S = H + (lam[:, None] if torch.is_tensor(lam) else lam) * F
    return (rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float)


ALL = SEL + CHK; tt = {k: np.digitize(NOV[D[k][3].numpy()], Q3) for k in ALL}; base = {k: hits(k, 1.) for k in ALL}
print('(1) oracle: r=3 top-5 of H + lam F relative to E2 (lam = 1), by novelty tercile of the target, folds %s' % ALL)
for lam in LAM:
    h = {k: hits(k, lam) for k in ALL}
    print('  lam %.1f: all %+6.1f %% | ' % (lam, 100 * (sum(h[k].sum() for k in ALL) / sum(base[k].sum() for k in ALL) - 1)) +
          ' | '.join('%s %+6.1f %%' % (nm, 100 * (sum(h[k][tt[k] == t].sum() for k in ALL) / sum(base[k][tt[k] == t].sum() for k in ALL) - 1)) for t, nm in enumerate(('novel', 'middle', 'familiar'))), flush=True)
print('(2) Spearman of per-trial features with the target\'s nearest-trained cosine (folds pooled):',
      '; '.join('%s %+.3f' % (f, spearmanr(torch.cat([D[k][4][f] for k in ALL]).numpy(), torch.cat([D[k][5] for k in ALL]).numpy())[0]) for f in D[ALL[0]][4]))
print('(3) routing by a per-trial feature: lam = lo below its median, hi above; chosen on %s, read on %s, vs E2' % (SEL, CHK))
for f in D[ALL[0]][4]:
    med = torch.cat([D[k][4][f] for k in SEL]).median(); best = (-1, None)
    for lo in (0., .5, 1., 1.5, 2.):
        for hi in (0., .5, 1., 1.5, 2.):
            s = sum(hits(k, torch.where(D[k][4][f] < med, lo, hi).float()).sum() for k in SEL) / sum(base[k].sum() for k in SEL)
            best = max(best, (s, (lo, hi)))
    lo, hi = best[1]; x = np.concatenate([hits(k, torch.where(D[k][4][f] < med, lo, hi).float()) for k in CHK]); b = np.concatenate([base[k] for k in CHK])
    bi = np.random.default_rng(0).integers(0, len(b), (500, len(b)))
    print('  %-14s lo %.1f hi %.1f: select %+.2f %% | check %+.2f ± %.2f %%' % (f, lo, hi, 100 * (best[0] - 1), 100 * (x.mean() / b.mean() - 1), 100 * (x[bi].mean(1) / b[bi].mean(1) - 1).std()), flush=True)
