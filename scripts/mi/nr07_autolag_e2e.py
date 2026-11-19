"""R07 end-to-end value of a label-free alignment step in front of the ensemble (sub16's composition on a fold).
The cohort's data arrive delayed by d samples (unknown). Per 64-trial batch, a detector member is run at the candidate
corrections c = -6..+6 and the one with the largest mean output norm is applied to the batch (R06); every member then
decodes the corrected batch. Held-out people, never-trained images: top-5 % at 5k (r=3, r=4) and 14k (r=3), the ensemble
as it is (shift d) against the ensemble behind the alignment step, paired; d = -4..+4, including d = 0 (its cost).
Variants of the estimator: raw argmax, argmax + 1 (the norm peaks one sample early on Alljoined people, R06).
    nr07_autolag_e2e.py <fold>"""
from nc_lib import *

k = int(sys.argv[1]); S = list(range(-5, 6)); tt = torch.arange(120)
MEM = {1: [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
           ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')],
       2: [('mi_d6_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'),
           ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')],  # fold 2 (the weakest people): the six lag-only members
       4: [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
           ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')],  # fold 4: sub16's composition without SQF (its fold-4 run comes later)
       3: [('mi_d6_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]}[k]  # fold 3: the lag-only members that exist there (no jitter: steeper under an offset)
X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); nb = len(X) // 64; n = nb * 64; X, subj, img = X[:n], subj[:n], img[:n]
p = f'{OUT}/autolag_e2e_f{k}.npy'
if not os.path.exists(p):
    E = np.zeros((len(S), n, 1536), np.float16)
    for recipe, ck in MEM:
        f = load(recipe, k, ck)
        for j, s in enumerate(S):
            E[j] += scaled(f(X[:, :, (tt - s).clamp(0, 119)], bs=1000).cpu().float()).numpy().astype(np.float16)
        print('cached', recipe, flush=True); del f; torch.cuda.empty_cache()
    np.save(p, E)
E = np.load(p, mmap_mode='r'); N = np.load(f'{OUT}/autolag_norm_f{k}_mi_rmix_ho5k_lag5.npy')[:nb]  # detector: rmix lag-only, shifts -8..+8
gals = {'5k': gallery(), '14k': gallery(big=True)}; groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]; bi = np.random.default_rng(0).integers(0, len(g4), (2000, len(g4)))


def hits(y):
    return {'5k r3': (rank(y[g4[:, :3]].mean(1), tg, gals['5k']) < 5).cpu().numpy(), '5k r4': (rank(y[g4].mean(1), tg, gals['5k']) < 5).cpu().numpy(), '14k r3': (rank(y[g4[:, :3]].mean(1), tg, gals['14k']) < 5).cpu().numpy()}


def assembled(tb):  # per batch b, the ensemble output at total shift tb[b] (clipped to the cached range)
    y = torch.empty(n, 1536); tb = np.clip(tb, S[0], S[-1])
    for s in np.unique(tb):
        rows = (np.nonzero(tb == s)[0][:, None] * 64 + np.arange(64)).ravel(); y[rows] = torch.as_tensor(np.asarray(E[S.index(s)][rows], dtype=np.float32))
    return y


zero = hits(torch.as_tensor(np.asarray(E[S.index(0)], dtype=np.float32))); print('ensemble at zero offset: ' + ' | '.join('%s %.3f' % (m, 100 * v.mean()) for m, v in zero.items()), flush=True)
def estimate(Nw, c, bias, parabola):
    """per batch: the shift to apply, from the detector norms Nw (batches, candidates c, evenly spaced)"""
    j = Nw.argmax(1)
    if not parabola:
        return c[j] + bias
    jj = np.clip(j, 1, len(c) - 2); r = np.arange(len(Nw)); l, m, h = Nw[r, jj - 1], Nw[r, jj], Nw[r, jj + 1]; cur = l - 2 * m + h
    v = c[jj] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1) * (c[1] - c[0]), 0.0)
    return np.round(np.clip(v, c[0], c[-1]) + bias).astype(int)


VAR = [('argmax of 13, +1', 1, 1, False), ('argmax of 13, +2', 1, 2, False), ('parabola, step 2, +1', 2, 1, True), ('parabola, step 2, +1.5', 2, 1.5, True), ('parabola, step 2, +2', 2, 2, True)]
t = np.arange(-8, 9)
for d in (int(a) for a in (sys.argv[2] if len(sys.argv) > 2 else '-4,-2,0,2,4').split(',')):
    base = hits(torch.as_tensor(np.asarray(E[S.index(d)], dtype=np.float32))); print('offset %+d | as is: %s' % (d, ' / '.join('%.3f' % (100 * base[m].mean()) for m in base)), flush=True)
    for name, step, bias, par in VAR:
        ok = (np.abs(t - d) <= 8) & ((t - d) % step == 0); tb = estimate(N[:, ok], t[ok], bias, par); h = hits(assembled(tb))  # candidates = corrections of -8..+8 around the received data, inside the cached range
        print('      %-24s %s (%s) residual %+.2f sd %.2f' % (name, ' / '.join('%.3f' % (100 * h[m].mean()) for m in h), ' / '.join('%+.1f +- %.1f %%' % (100 * (h[m].mean() / base[m].mean() - 1), 100 * (h[m][bi].mean(1) / base[m][bi].mean(1) - 1).std()) for m in h), tb.mean(), tb.std()), flush=True)
