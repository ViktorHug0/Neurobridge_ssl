"""Y11 why single-trial decoding leads the board but the repeat average lags (board, 18 Nov: per-trial top-1 1st; aggregation gain =
subj-agg top-5 / per-trial top-5 = 1.94 for finalA, 2.00 for sub16, 2.03 for jmokim). The grader averages RAW outputs over the ~3
repeats of a (person, image), so each trial's norm is its weight in that average; the per-trial metrics ignore the norm.
Held-out people x 5,000 never-trained images, 5k gallery, cached fold-model outputs (u01), four leave-one-out triplets per group.
(1) per member and for the six / nine: per-trial top-1 / top-5, r=3 top-5, gain r3 / r1;
(2) per-trial confidence features of the nine (norm, member agreement, peak against the training-image bank): AUC for a per-trial hit;
(3) the nine's direction with trial weight f = norm^a x agreement^b x bank-peak^g before the repeat average; r=3 top-5 against the
    deployed raw sum (a=1, b=g=0), paired.
    ny11_agg_gain.py <folds> [name=recipe:checkpoint ...]   (SOLO=1: part 1 only)"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]
MEM = {'d6': 'mi_d6_ho5k_lag5_jit3:checkpoint_test_best.pth', 'rmix': 'mi_rmix_ho5k_lag5_jit3:checkpoint_ep10.pth',
       'SQF-X': 'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth', 'REVE-scr': 'mi_scr_ho5k_lag5:checkpoint_test_best.pth',
       'SQF': 'mi_sqf_ho5k_lag5:checkpoint_ep32.pth', 'LaBraM long': 'mi_plab_ho5k_lag5_long:checkpoint_ep10.pth',
       'LaBraM': 'mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth', 'rmix soft': 'mi_rmix_ho5k_lag5_jit3_soft3:checkpoint_ep10.pth',
       'LaBraM soft': 'mi_plab_ho5k_lag5_long_soft3:checkpoint_ep10.pth', 'ST soft': 'mi_stf_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth',
       'ST': 'mi_stf_ho5k_lag5_jit3:checkpoint_test_best.pth', 'REVE-pre soft': 'mi_prev_ho5k_lag5_jit3_soft3:checkpoint_test_best.pth'}
MEM.update(a.split('=', 1) for a in sys.argv[2:])  # extra members: name=recipe:checkpoint (folds without the cache are skipped for them)
SIX = ['d6', 'rmix', 'SQF-X', 'REVE-scr', 'SQF', 'LaBraM long']
NINE = ['SQF-X', 'REVE-scr', 'SQF', 'LaBraM', 'rmix soft', 'LaBraM soft', 'ST soft', 'ST', 'REVE-pre soft']
gal = gallery(); bank = nz(torch.as_tensor(TARGETS[np.setdiff1d(np.arange(16540), HELD)]).float()).to(DEV)  # training images only
GRID = [(a, b, g) for a in (0., .5, 1., 1.5) for b in (0., 1., 2., 4.) for g in (0., 2., 4.)]
R1, R3, AUC, GH = {}, {}, {}, {}


def auc(s, h):  # P(score of a hit > score of a miss)
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = h.sum(); return (r[h].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(s) - n1))


for k in KS:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); T = np.unique(g4); loc = np.searchsorted(T, g4)  # trials used, group -> rows
    g3 = torch.as_tensor(np.concatenate([loc[:, [a for a in range(4) if a != o]] for o in range(4)])); tg3 = torch.as_tensor(img)[T][g3[:, 0]]
    t1 = torch.as_tensor(img)[T]; P = {}
    for name, rc in MEM.items():
        r, c = rc.split(':')
        if not os.path.exists(f'{OUT}/u01_f{k}_{r}_{c}.npy'):
            continue
        if k == 2 and name == 'LaBraM soft':
            r, c = 'mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'
        P[name] = scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{r}_{c}.npy').astype(np.float32)))[T].half()
    def ev(S):  # per-trial ranks, r=3 hits
        return rank(S, t1, gal).numpy(), (rank(S[g3].mean(1), tg3, gal) < 5).numpy().astype(float)
    for name in [m for m in MEM if m in P] + ['six', 'nine']:
        S = sum(P[m].float() for m in (SIX if name == 'six' else NINE)) if name in ('six', 'nine') else P[name].float()
        r1, h3 = ev(S); R1.setdefault(name, []).append(r1); R3.setdefault(name, []).append(h3)
    if os.environ.get('SOLO'):  # part (1) only
        del P; print('fold', k, 'done', flush=True); continue
    S = sum(P[m].float() for m in NINE); n = S.norm(dim=1); u = sum(nz(P[m].float()) for m in NINE); A = u.norm(dim=1) / len(NINE)
    cb = torch.cat([(nz(S[i:i + 4096]).to(DEV) @ bank.T).topk(2, dim=1).values.cpu() for i in range(0, len(S), 4096)])
    F = {'norm': n, 'agreement': A, 'bank peak': cb[:, 0], 'bank margin': cb[:, 0] - cb[:, 1]}
    hit = R1['nine'][-1] < 5
    for f, v in F.items():
        AUC.setdefault(f, []).append(auc(v.numpy(), hit))
    d = nz(S)
    for a, b, g in GRID:
        f = n ** a * A ** b * (cb[:, 0] / cb[:, 0].mean()) ** g
        GH.setdefault((a, b, g), []).append((rank((d * f[:, None])[g3].mean(1), tg3, gal) < 5).numpy().astype(float))
    del P, S, u, d; print('fold', k, 'done', flush=True)

print('(1) folds %s, 5k gallery: per-trial top-1 / top-5, r=3 top-5 (four triplets), gain r3/r1 top-5' % KS)
for name in R1:
    r1 = np.concatenate(R1[name]); h3 = np.concatenate(R3[name])
    print('  %-14s top-1 %.3f %%  top-5 %.3f %%  | r=3 top-5 %.3f %% | gain %.2f | per fold %s' % (name, 100 * (r1 < 1).mean(), 100 * (r1 < 5).mean(), 100 * h3.mean(),
          h3.mean() / (r1 < 5).mean(), ' '.join('%.2f' % (b.mean() / (a < 5).mean()) for a, b in zip(R1[name], R3[name]))), flush=True)
if os.environ.get('SOLO'):
    sys.exit()
print('(2) AUC of the nine\'s per-trial features for a per-trial top-5 hit, per fold:', '; '.join('%s %s' % (f, ' '.join('%.3f' % x for x in v)) for f, v in AUC.items()))
print('(3) trial weight norm^a x agreement^b x bank-peak^g on the nine\'s direction, r=3 top-5 vs deployed raw sum (a=1, b=g=0), paired')
base = GH[1., 0., 0.]; rows = []
for key, hs in GH.items():
    a = np.concatenate(hs); b = np.concatenate(base); bi = np.random.default_rng(0).integers(0, len(b), (500, len(b)))
    rows.append((100 * (a.mean() / b.mean() - 1), key, 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(), ' '.join('%+.1f' % (100 * (x.mean() / y.mean() - 1)) for x, y in zip(hs, base))))
for v, key, sd, pf in sorted(rows, reverse=True)[:12] + [r for r in rows if r[1] == (0., 0., 0.)]:
    print('  a %.1f b %.0f g %.0f: %+.2f ± %.2f %% | per fold %s' % (*key, v, sd, pf), flush=True)
