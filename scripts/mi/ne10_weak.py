"""E10 do the levers look the same for weak and strong new people? The sealed set is ~2x harder than the proxy (E00) and
the pooled proxy is carried by its best people (E05). Cached clean-proxy outputs, folds 1-3, 5k, exact ranks, r=3 (r=1 in
brackets); people split at the median of their ens5 score: weak = sub-04, 07, 08, 11, 15, 16, 17, 19 (0.2-1.1 %),
strong = the other seven. Per stratum: members alone, ensemble composition, REVE budget, lag swaps."""
import pandas as pd
from nc_lib import *

WEAK = {4, 7, 8, 11, 15, 16, 17, 19}
B = {1: dict(d6='reve_d6_ho5k_long_optx_things', rmix='rmix_reg_ho5k_long_optx_things', scr='reve_scratch_ho5k_long_optx_things'),
     2: dict(d6='reve_d6_ho5k_mid_optx_things', rmix='rmix_reg_ho5k_mid_optx_things', scr='reve_scratch_ho5k_long_optx_things'),
     3: dict(d6='reve_d6_ho5k_mid_optx_things', rmix='rmix_reg_ho5k_mid_optx_things', scr='reve_scratch_ho5k_mid_optx_things')}
gal = gallery()
H = {}  # (set name, stratum, r) -> list of (hits, base hits) per fold


def load_y(k, n):
    F = f'{R}/logs/holdout/f{k}'
    if n in ('sqfx', 'sqf'):
        p = f'{F}/' + ('sqf_dbn_ho5k_long_optx_things.npy' if n == 'sqfx' else 'sqf_dbn_ho5k_long_things__ep32.npy')
    elif 'L' in n:  # lag version: d6L@10, rmixL@10, sqfxL
        r, e = (n[:-1], 40) if n == 'sqfxL' else (n.split('L@')[0], int(n.split('@')[1]))
        p = f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy'
    else:
        m, e = n.split('@'); p = f'{F}/{B[k][m]}__ep{int(e):02d}.npy'
        if not os.path.exists(p) and json.load(open(sorted(glob.glob(f'{ROOT}/{B[k][m]}/fold{k}/*/*/train_config.json'))[-1]))['stop_after_epoch'] == int(e):
            p = f'{F}/{B[k][m]}.npy'
    return scaled(torch.as_tensor(np.load(p)).float()) if os.path.exists(p) else None


E5 = ['d6@10', 'rmix@10', 'sqfx', 'scr@10', 'sqf']
SETS = {'ens5 (REVE@10)': E5, **{f'{m} alone': [m] for m in E5}, 'REVE trio': ['d6@10', 'rmix@10', 'scr@10'], 'SQF pair': ['sqfx', 'sqf'],
        **{f'ens5 without {m}': [x for x in E5 if x != m] for m in E5},
        'ens5 REVE@8': ['d6@8', 'rmix@8', 'sqfx', 'scr@8', 'sqf'], 'ens5 REVE@12': ['d6@12', 'rmix@12', 'sqfx', 'scr@12', 'sqf'],
        'lag: rmix': ['d6@10', 'rmixL@10', 'sqfx', 'scr@10', 'sqf'], 'lag: SQF-X': ['d6@10', 'rmix@10', 'sqfxL', 'scr@10', 'sqf'], 'lag: d6': ['d6L@10', 'rmix@10', 'sqfx', 'scr@10', 'sqf'],
        'lag: rmix + SQF-X': ['d6@10', 'rmixL@10', 'sqfxL', 'scr@10', 'sqf'], 'lag: d6 + rmix + SQF-X': ['d6L@10', 'rmixL@10', 'sqfxL', 'scr@10', 'sqf']}
for k in (1, 2, 3):
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
    cache = {}
    def y(names):
        for n in names:
            if n not in cache:
                cache[n] = load_y(k, n)
        return None if any(cache[n] is None for n in names) else sum(cache[n] for n in names)
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])); t3 = torch.as_tensor(img)[g3[:, 0]]; s3 = subj[g3[:, 0].numpy()]
    w3, w1 = np.isin(s3, list(WEAK)), np.isin(subj, list(WEAK))
    def hit(yy):
        return {3: (rank(yy[g3].mean(1), t3, gal) < 5).numpy(), 1: (rank(yy, torch.as_tensor(img), gal) < 5).numpy()}
    hb = hit(y(E5))
    for n, names in SETS.items():
        yy = y(names)
        if yy is None:
            continue
        h = hit(yy)
        for r, w in ((3, w3), (1, w1)):
            for sn, mask in (('weak', w), ('strong', ~w)):
                H.setdefault((n, sn, r), []).append((h[r][mask], hb[r][mask], k))
    print('fold', k, flush=True)
rows = []
for n in SETS:
    if (n, 'weak', 3) not in H:
        continue
    row = dict(set=n, folds=''.join(str(v[2]) for v in H[(n, 'weak', 3)]))
    for sn in ('weak', 'strong'):
        for r in (3, 1):
            a = np.concatenate([v[0] for v in H[(n, sn, r)]]); b = np.concatenate([v[1] for v in H[(n, sn, r)]])
            row[f'{sn} r{r}'] = '%.3f %+.1f +- %.1f' % (100 * a.mean(), *paired(a, b))
    rows.append(row)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
