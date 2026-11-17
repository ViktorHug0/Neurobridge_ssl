"""C01 does the clean proxy reproduce the SHAPE of our sealed leaderboard numbers? The sub-2 ensemble on the proxy
(the five --holdout_images runs at sub-2 budgets, scaled sum; cached outputs of logs/holdout/f<k>; new people x
never-trained images), exact target ranks in the 14,366-image clean gallery (5k held-out + 9.4k other THINGS images).
Accuracy at any smaller gallery G' follows exactly from the ranks (hypergeometric); 40k is a log-log extrapolation from
G' = 3.5k..14k. Printed per G' and repeat count r: top-1, top-5, their ratio, and the gain over r=1 (per-epoch).
Sealed board, sub 2: per-epoch top-5 .0554 %, top-1 .0142 %; subject-aggregated top-5 .1054 %, top-1 .0267 %.
    nc08_calibrate.py <folds e.g. 1,2,3>"""
import pandas as pd
from scipy.stats import hypergeom
from nc_lib import *

folds = [int(a) for a in sys.argv[1].split(',')]
MEM = {1: ('reve_d6_ho5k_long_optx_things__ep08', 'rmix_reg_ho5k_long_optx_things__ep08', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep08', 'sqf_dbn_ho5k_long_things__ep32'),
       2: ('reve_d6_ho5k_mid_optx_things__ep08', 'rmix_reg_ho5k_mid_optx_things__ep08', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep08', 'sqf_dbn_ho5k_long_things__ep32'),
       3: ('reve_d6_ho5k_mid_optx_things__ep08', 'rmix_reg_ho5k_mid_optx_things__ep08', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_mid_optx_things__ep08', 'sqf_dbn_ho5k_long_things__ep32')}
SETS = {'sub-2 ensemble': (0, 1, 2, 3, 4), 'rmix': (1,), 'SQF-X': (2,), 'REVE trio': (0, 1, 3)}
gal = gallery(big=True); G = len(gal)
GS = [1000, 2000, 3500, 5000, 7000, 10000, G]
counts = {}  # (set, r) -> histogram of target ranks over all folds
for k in folds:
    F = f'{R}/logs/holdout/f{k}'
    st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    Y = [scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in MEM[k]]
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4]))
    t1 = torch.as_tensor(img); t4 = t1[g4[:, 0]]
    for name, idx in SETS.items():
        y = sum(Y[i] for i in idx)
        for r in (1, 2, 3, 4):
            rk = rank(y, t1, gal) if r == 1 else rank(y[g4[:, :r]].mean(1), t4, gal)
            counts[(name, r)] = counts.get((name, r), 0) + np.bincount(rk.numpy(), minlength=G)
    print('fold', k, 'done', flush=True)


def acc(h, Gp, k):  # P(target in the top k of a random sub-gallery of size Gp that contains it), in %
    return 100 * (h * hypergeom.cdf(k - 1, G - 1, np.arange(G), Gp - 1)).sum() / h.sum()


pd.set_option('display.width', 250)
for name in SETS:
    rows = []
    for Gp in GS + [40000]:
        row = dict(G=Gp)
        for r in (1, 2, 3, 4):
            for k in (1, 5):
                if Gp <= G:
                    row[f'top{k} r{r}'] = acc(counts[(name, r)], Gp, k)
                else:  # log-log extrapolation from the three largest galleries
                    x = np.log([GS[-3], GS[-2], GS[-1]]); yv = np.log([acc(counts[(name, r)], g, k) for g in GS[-3:]])
                    row[f'top{k} r{r}'] = float(np.exp(np.polyval(np.polyfit(x, yv, 1), np.log(Gp))))
        rows.append(row)
    df = pd.DataFrame(rows).set_index('G')
    shape = pd.DataFrame({'top5/top1 r1': df['top5 r1'] / df['top1 r1'], **{f'top5/top1 r{r}': df[f'top5 r{r}'] / df[f'top1 r{r}'] for r in (2, 3, 4)},
                          **{f'agg gain top5 r{r}': df[f'top5 r{r}'] / df['top5 r1'] for r in (2, 3, 4)}, **{f'agg gain top1 r{r}': df[f'top1 r{r}'] / df['top1 r1'] for r in (2, 3, 4)}})
    print(f'\n== {name}: accuracy % (last row = extrapolated)'); print(df.round(4).to_string())
    print('shape (sealed sub 2: top5/top1 3.90 per-epoch, 3.95 aggregated; aggregation gain 1.90 top-5, 1.88 top-1)'); print(shape.round(2).to_string())
    df.join(shape).to_csv(f'{OUT}/nc08_calibrate_{name.replace(" ", "_")}.csv')
