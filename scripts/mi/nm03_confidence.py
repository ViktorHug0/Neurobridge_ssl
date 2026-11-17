"""M03 can a trial's quality be predicted, to weight the repeats better than the output norm does?
The grader averages raw outputs within (person, image): a trial's output norm is its weight (E01: exponent 1 is the best
power of the norm). Here a regressor predicts each trial's quality q = cosine(ensemble output, its target) from
label-free features: member norms, agreement of each member with the rest, and input statistics (per-electrode RMS,
peak, prestimulus RMS, band powers). Fitted on the new people of two folds, applied to the third (other people, other
models); the trial is re-weighted by exp(b * predicted q, standardised). ens5 (sub-6 budgets, cached), never-trained
images, 5k gallery, r=3, exact ranks, folds pooled, paired against the plain scaled sum. Oracle = the true q."""
from sklearn.ensemble import HistGradientBoostingRegressor
from nc_lib import *

gal = gallery(); D = {}
for k in (1, 2, 3):
    F = f'{R}/logs/holdout/f{k}'
    st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    Y = [scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in ENS5[k]]; y = sum(Y)
    X = torch.as_tensor(np.load(f'{F}/X.npy')).float()
    P = torch.fft.rfft(X, dim=-1).abs().pow(2).mean(1)  # (n, 61) power per 1 Hz bin, mean over electrodes
    band = lambda a, b: P[:, a:b].sum(1).clamp_min(1e-9).log()
    feats = [y.norm(dim=1).log()] + [m.norm(dim=1).log() for m in Y] + [(nz(m) * nz(y - m)).sum(1) for m in Y] + \
            [X.pow(2).mean((1, 2)).log() / 2, X.abs().amax((1, 2)).log(), X[..., :24].pow(2).mean((1, 2)).log() / 2, band(1, 4), band(4, 8), band(8, 13), band(13, 30), band(30, 61)] + \
            list((X.pow(2).mean(2).clamp_min(1e-9).log() / 2).T)
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    D[k] = dict(f=torch.stack(feats, 1).numpy(), q=(nz(y) * gal[torch.as_tensor(img)].cpu()).sum(1).numpy(), y=y, img=torch.as_tensor(img),
                g3=torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])))
    print('fold', k, D[k]['f'].shape, flush=True); del Y, X, P
FN = ['|sum|'] + [f'|{m}|' for m in ('d6', 'rmix', 'sqfx', 'scr', 'sqf')] + [f'agree {m}' for m in ('d6', 'rmix', 'sqfx', 'scr', 'sqf')] + \
     ['rms', 'peak', 'prestim rms', '1-4 Hz', '4-8 Hz', '8-13 Hz', '13-30 Hz', '30-60 Hz'] + [f'rms e{c}' for c in range(32)]
SETS = {'output only (norms + agreement)': list(range(11)), 'input only': list(range(11, len(FN))), 'all': list(range(len(FN)))}
hits = lambda d, w: (rank((d['y'] * torch.as_tensor(w).float()[:, None])[d['g3']].mean(1), d['img'][d['g3'][:, 0]], gal) < 5).numpy()
base = {k: hits(D[k], np.ones(len(D[k]['q']))) for k in D}
show = lambda n, h: print('  %-46s %.3f   %+.1f +- %.1f %%' % (n, 100 * np.concatenate(list(h.values())).mean(), *paired(np.concatenate(list(h.values())), np.concatenate(list(base.values())))), flush=True)
z = lambda v: (v - v.mean()) / v.std()
show('plain scaled sum', base)
for b in (.25, .5, 1.):
    show(f'oracle: weight exp({b} z(true q))', {k: hits(D[k], np.exp(b * z(D[k]['q']))) for k in D})
for sn, cols in SETS.items():
    pred = {}
    for k in D:
        tr = [j for j in D if j != k]
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., random_state=0)
        gb.fit(np.concatenate([D[j]['f'][:, cols] for j in tr]), np.concatenate([D[j]['q'] for j in tr]))
        pred[k] = gb.predict(D[k]['f'][:, cols])
    print(f'{sn}: correlation predicted-true q on the left-out fold:', ' '.join('%.3f' % np.corrcoef(pred[k], D[k]['q'])[0, 1] for k in D),
          '| |sum| alone:', ' '.join('%.3f' % np.corrcoef(D[k]['f'][:, 0], D[k]['q'])[0, 1] for k in D), flush=True)
    for b in (.1, .2, .4):
        show(f'{sn}: weight exp({b} z(pred q))', {k: hits(D[k], np.exp(b * z(pred[k]))) for k in D})
    for b in (.1, .2, .4):  # same, with the norm's own weight removed first: the predicted quality replaces the norm
        show(f'{sn}: unit norm, weight exp({b} z(pred q))', {k: hits(D[k], np.exp(b * z(pred[k])) / D[k]['y'].norm(dim=1).numpy()) for k in D})
