"""E09 new-CONCEPT proxy from existing fold models: the 200 official test images (concepts absent from training; the cv4
ho5k runs never train on them) x the fold's held-out people x 80 repeats. Queries = r=3 (26 per person-image) and r=4
(20) groups of repeats, 64-trial batches mixing people, scaled sum; gallery = the 200 test targets + the 14,367
never-trained training-concept images (14.6k) or + the 5k held-out ones (5.2k). Exact ranks; paired bootstrap over
(person, image) clusters. Checks on new concepts: the lag members, the REVE epoch budget (8 / 10 / 12).
    ne09_newconcept.py <fold>"""
import pandas as pd
from nc_lib import *

k = int(sys.argv[1])
L, M = ('long', 'mid')[k != 1], {}
for e in (8, 10, 12):
    ck = f'checkpoint_ep{e:02d}.pth'
    M[f'd6@{e}'] = (f'reve_d6_ho5k_{L}_optx_things', ck); M[f'rmix@{e}'] = (f'rmix_reg_ho5k_{L}_optx_things', ck); M[f'scr@{e}'] = (f'reve_scratch_ho5k_{L}_optx_things', ck)
    M[f'd6L@{e}'] = ('mi_d6_ho5k_lag5', ck); M[f'rmixL@{e}'] = ('mi_rmix_ho5k_lag5', ck)
M.update({'sqfx': ('sqf_dbn_ho5k_long_optx_things', 'checkpoint_test_best.pth'), 'sqf': ('sqf_dbn_ho5k_long_things', 'checkpoint_ep32.pth'), 'sqfxL': ('mi_sqfx_ho5k_lag5', 'checkpoint_test_best.pth')})


def ckpt(recipe, ck):  # a missing checkpoint_epNN = the run's final epoch if its stop epoch is NN
    d = sorted(glob.glob(f'{ROOT}/{recipe}/fold{k}/*/*/train_config.json'))
    if not d:
        return None
    d = os.path.dirname(d[-1])
    if os.path.exists(f'{d}/{ck}'):
        return ck
    stop = json.load(open(f'{d}/train_config.json'))['stop_after_epoch']
    return 'checkpoint_test_best.pth' if ck == f'checkpoint_ep{stop:02d}.pth' and os.path.exists(f'{d}/track1_score.json') else None


X = np.concatenate([np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r')[:, 0].reshape(-1, 32, 120) for s in FOLDS[k]])
subj = np.repeat(FOLDS[k], 16000); img = np.tile(np.repeat(np.arange(200), 80), 5); rep = np.tile(np.arange(80), 1000)
o = np.random.default_rng(11).permutation(len(X)); X, subj, img, rep = X[o], subj[o], img[o], rep[o]
pos = np.empty(len(o), int); pos[np.lexsort((rep, img, subj))] = np.arange(len(o))  # rank of each trial inside its (person, image) block
block = np.argsort(pos).reshape(1000, 80)  # trial indices per (person, image), repeats in order
G = {r: torch.as_tensor(block[:, :80 // r * r].reshape(-1, r)) for r in (3, 4)}
tgt = {r: torch.as_tensor(img[G[r][:, 0].numpy()]) for r in G}; clu = {r: np.repeat(np.arange(1000), 80 // r) for r in G}
T = nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/image_test.npy').reshape(200, 1536)).float()).to(DEV)
gals = {'14.6k': torch.cat([T, gallery(big=True)]), '5.2k': torch.cat([T, gallery()])}
Y = {}
for n, (recipe, ck) in M.items():
    c = ckpt(recipe, ck)
    if c is None:
        continue
    p = f'{OUT}/test_f{k}_{recipe}_{c[11:-4]}.npy'
    if not os.path.exists(p):
        np.save(p, load(recipe, k, c)(X).numpy().astype(np.float16)); torch.cuda.empty_cache()
    Y[n] = scaled(torch.as_tensor(np.load(p)).float())
print('members:', ' '.join(Y), flush=True)


def hits(names):
    y = sum(Y[n] for n in names)
    return {(g, r): np.bincount(clu[r], (rank(y[G[r]].mean(1), tgt[r], gal) < 5).numpy(), 1000) / (80 // r) for g, gal in gals.items() for r in G}  # per-cluster hit rate


def cmp(a, b, n=2000):
    i = np.random.default_rng(0).integers(0, 1000, (n, 1000)); return 100 * (a.mean() / b.mean() - 1), (100 * (a[i].mean(1) / b[i].mean(1) - 1)).std()


rows = []
def show(title, sets, ref):
    hb = hits(sets[ref])
    for n, names in sets.items():
        if not all(m in Y for m in names):
            continue
        h = hits(names)
        rows.append(dict(block=title, set=n, **{f'{g} r{r}': '%.3f (%+.1f +- %.1f)' % (100 * h[(g, r)].mean(), *cmp(h[(g, r)], hb[(g, r)])) for g in gals for r in G}))
        print(rows[-1], flush=True)


show('singles', {n: [n] for n in Y}, 'd6@10')
e5 = lambda e, d6='d6', rm='rmix', sx='sqfx': [f'{d6}@{e}', f'{rm}@{e}', sx, f'scr@{e}' if f'scr@{e}' in Y else 'scr@10', 'sqf']
show('ens5 by REVE budget', {f'ens5 REVE@{e}': e5(e) for e in (8, 10, 12)}, 'ens5 REVE@10')
show('lag swaps at budget 10', {'ens5 REVE@10': e5(10), 'rmix lag': e5(10, rm='rmixL'), 'SQF-X lag': e5(10, sx='sqfxL'), 'd6 lag': e5(10, d6='d6L'), 'rmix + SQF-X lag': e5(10, rm='rmixL', sx='sqfxL'),
                                    'd6 + rmix + SQF-X lag': e5(10, 'd6L', 'rmixL', 'sqfxL')}, 'ens5 REVE@10')
show('all lag by REVE budget', {f'lag ens5 REVE@{e}': e5(e, 'd6L' if 'd6L@10' in Y else 'd6', 'rmixL', 'sqfxL') for e in (8, 10, 12)}, 'lag ens5 REVE@10')
pd.DataFrame(rows).to_csv(f'{OUT}/e09_newconcept_f{k}.csv', index=False)
