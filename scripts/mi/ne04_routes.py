"""E04 a committee of person routes. The members decode conditionally on the person's mean ERP (key): each training
person has its own route. A new trial is sent through several training people's routes by adding (k_t - K) to it
(K = mean key of the Alljoined training people; stateless, deployable) and the outputs are averaged, with or without
the unmodified trial. Are foreign routes diverse enough to act as extra ensemble members?
Fold k, new people x never-trained images, 5k, exact ranks, paired against the plain member.
    ne04_routes.py <fold> <members> [scale of the added offset, default 1]"""
import pandas as pd
from nc_lib import *

k = int(sys.argv[1]); members = sys.argv[2].split(','); a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.
NEW = FOLDS[k]; TRAIN = [s for s in range(1, 21) if s not in NEW]
F = f'{R}/logs/holdout/f{k}'
st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']; X = np.load(f'{F}/X.npy', mmap_mode='r')
gal = gallery()
keys = {s: key(s).to(DEV) for s in TRAIN}; K = torch.stack(list(keys.values())).mean(0)
order = [TRAIN[i] for i in np.random.default_rng(0).permutation(len(TRAIN))]
rows, outs = [], {}
for m in members:
    f = load(m, k)
    raw = scaled(f(X)); route = [scaled(f(X, lambda b, e, t=t: a * (keys[t] - K))) for t in order[:8]]
    outs[m] = {'raw': raw, 'one route (mean of 4 single routes)': None, 'mean of 2 routes': sum(route[:2]) / 2, 'mean of 4 routes': sum(route[:4]) / 4, 'mean of 8 routes': sum(route) / 8,
               'raw + mean of 4 routes': raw + sum(route[:4]) / 4, 'raw + mean of 8 routes': raw + sum(route) / 8, 'raw + .5 mean of 8 routes': raw + .5 * sum(route) / 8}
    b = score(raw, subj, img, gal, hits=True)
    single = [score(r, subj, img, gal) for r in route[:4]]
    print(dict(member=m, cond='raw', **b[0]), flush=True)
    print(dict(member=m, cond='single routes r3', r3=[round(s['r3'], 3) for s in single], cos_raw_route=float((nz(raw[:20000]) * nz(route[0][:20000])).sum(1).mean()),
               cos_route_route=float((nz(route[1][:20000]) * nz(route[0][:20000])).sum(1).mean())), flush=True)
    for cn, y in outs[m].items():
        if y is None or cn == 'raw':
            continue
        sc, h = score(y, subj, img, gal, hits=True)
        rows.append(dict(member=m, cond=cn, **sc, gain_r3='%+.1f +- %.1f %%' % paired(h['r3'], b[1]['r3']), gain_r4='%+.1f +- %.1f %%' % paired(h['r4'], b[1]['r4'])))
        print(rows[-1], flush=True)
    del f, route; torch.cuda.empty_cache()
if len(members) > 1:
    b = score(sum(outs[m]['raw'] for m in members), subj, img, gal, hits=True)
    print(dict(member='+'.join(members), cond='raw', **b[0]))
    for cn in outs[members[0]]:
        if outs[members[0]][cn] is None or cn == 'raw':
            continue
        sc, h = score(sum(outs[m][cn] for m in members), subj, img, gal, hits=True)
        rows.append(dict(member='+'.join(members), cond=cn, **sc, gain_r3='%+.1f +- %.1f %%' % paired(h['r3'], b[1]['r3']), gain_r4='%+.1f +- %.1f %%' % paired(h['r4'], b[1]['r4'])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
