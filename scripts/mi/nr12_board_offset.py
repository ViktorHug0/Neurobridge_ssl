"""R12 which sealed latency offset are our two board scores most consistent with?
Board: sub 2 (five members, no lag correction, no jitter) per-epoch top-5 0.0554 %, aggregated 0.1054 %; sub16 (lag + jitter
five + LaBraM) 0.0629 % / 0.1258 %: ratios 1.135 (single trial) and 1.194 (aggregated, ~2.5-3 repeats).
Fold 1, held-out people, never-trained images: the sub-2 composition (nc_lib MEMBERS, pre-lag recipes at epoch 10) and the
sub16 composition (cache of nr07) under a common delay d: predicted ratio sub16 / sub 2 at r=1 and r=3 (5k and 14k).
    nr12_board_offset.py [fold, default 1]"""
from nc_lib import *

k = int(sys.argv[1]) if len(sys.argv) > 1 else 1; S = list(range(-5, 6)); tt = torch.arange(120); X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); n = len(X) // 64 * 64; X, subj, img = X[:n], subj[:n], img[:n]
p = f'{OUT}/autolag_sub2_f{k}.npy'
if not os.path.exists(p):
    E = np.zeros((len(S), n, 1536), np.float16)
    for m in ('d6', 'rmix', 'sqfx', 'sqf', 'scr') if k == 1 else ('d6m', 'rmixm', 'sqfx', 'sqf', 'scr' if k == 2 else 'reve_scratch_ho5k_mid_optx_things'):  # folds 2-4: the 'mid' REVE runs (fold 3: REVE-scratch too, as ENS5)
        f = load(m, k)
        for j, s in enumerate(S):
            E[j] += scaled(f(X[:, :, (tt - s).clamp(0, 119)], bs=1000).cpu().float()).numpy().astype(np.float16)
        print('cached', m, MEMBERS.get(m, m), flush=True); del f; torch.cuda.empty_cache()
    np.save(p, E)
q = f'{OUT}/autolag_e2e_f{k}.npy' if k == 1 else f'{OUT}/autolag_s16_f{k}.npy'  # fold 1: nr07's cache is the sub16 composition; other folds: built here
if not os.path.exists(q):
    E = np.zeros((len(S), n, 1536), np.float16)
    for recipe, ck in (('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
                       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')):
        f = load(recipe, k, ck)
        for j, s in enumerate(S):
            E[j] += scaled(f(X[:, :, (tt - s).clamp(0, 119)], bs=1000).cpu().float()).numpy().astype(np.float16)
        print('cached', recipe, flush=True); del f; torch.cuda.empty_cache()
    np.save(q, E)
A, B = np.load(p, mmap_mode='r'), np.load(q, mmap_mode='r'); gals = {'5k': gallery(), '14k': gallery(big=True)}; groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]; t1 = torch.as_tensor(img); bi = np.random.default_rng(0).integers(0, len(g4), (1000, len(g4))); b1 = np.random.default_rng(1).integers(0, n, (1000, n))


def hits(E, s, g):
    y = torch.as_tensor(np.asarray(E[S.index(s)], dtype=np.float32)); return (rank(y, t1, gals[g]) < 5).cpu().numpy(), (rank(y[g4[:, :3]].mean(1), tg, gals[g]) < 5).cpu().numpy()


print('board: single trial x1.135, aggregated x1.194')
for d in range(-5, 6):
    out = []
    for g in gals:
        a1, a3 = hits(A, d, g); c1, c3 = hits(B, d, g)
        out.append('%s r=1 x%.3f +- %.3f, r=3 x%.3f +- %.3f' % (g, c1.mean() / a1.mean(), (c1[b1].mean(1) / a1[b1].mean(1)).std(), c3.mean() / a3.mean(), (c3[bi].mean(1) / a3[bi].mean(1)).std()))
    print('delay %+d | predicted sub16 / sub 2: %s' % (d, ' | '.join(out)), flush=True)
