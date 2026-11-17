"""E12 concept novelty vs test-run effect (fold 1, REVE-d6 ep10, exact ranks, r=1 / r=3 top-5 %).
 (a) concept-holdout model on its 500 held-out concepts (training-split runs, 5k gallery of those images)
 (c) image-holdout model on its 5,000 held-out images of trained concepts (5k gallery)
 (a') / (c') the same two with a matched gallery structure: queries restricted to 3 images per concept x the 500 held-out
      concepts; gallery = the 1,500 such images + the 9.4k other THINGS images
 (b) / (d) both models on the 200 test images (new concepts for both, test runs), gallery = 200 + 5k held-out images"""
from nc_lib import *
HC = np.sort((np.random.default_rng(0).choice(1654, 500, replace=False)[:, None] * 10 + np.arange(10)).ravel())
extra = gallery(big=True)[5000:]
def load_stream(d):
    st = np.load(f'{d}/stream.npz'); return st['subj'], st['img']
def sc(y, subj, img, gal, keep=None):
    if keep is not None:
        y, subj, img = y[torch.as_tensor(keep)], subj[keep], img[keep]
    s = score(y, subj, img, gal); return '%.3f / %.3f' % (s['r1'], s['r3'])
Yc = torch.as_tensor(np.load(f'{R}/logs/holdout_concepts/f1/reve_d6_hoc500_long_optx_things__ep10.npy')).float(); sc_, ic_ = load_stream(f'{R}/logs/holdout_concepts/f1')
Yi = torch.as_tensor(np.load(f'{R}/logs/holdout/f1/reve_d6_ho5k_long_optx_things__ep10.npy')).float(); si_, ii_ = load_stream(f'{R}/logs/holdout/f1')
print('(a) new concepts, training runs (concept-holdout model, its 5k):', sc(Yc, sc_, ic_, gallery(HC)))
print('(c) trained concepts, new images (image-holdout model, its 5k): ', sc(Yi, si_, ii_, gallery(HELD)))
# matched structure: the 500 held-out concepts, 3 images each
rng = np.random.default_rng(1)
pickc = np.sort(np.concatenate([c * 10 + rng.choice(10, 3, replace=False) for c in range(500)]))  # positions in HC (concept-major)
mapc = np.full(5000, -1); mapc[pickc] = np.arange(1500)
keep = mapc[ic_] >= 0
yk = scaled(Yc)[torch.as_tensor(keep)]; s = score.__wrapped__ if hasattr(score, '__wrapped__') else None
def sc2(Y, subj, img, keep, newidx, gal):
    # scaled on the full stream first (64-batches as served), then restricted
    y = scaled(Y)[torch.as_tensor(keep)]; subj, img = subj[keep], newidx[img[keep]]
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 3])); t = torch.as_tensor(img)
    return '%.3f / %.3f  (%d groups)' % (100 * (rank(y, t, gal) < 5).float().mean(), 100 * (rank(y[g].mean(1), t[g[:, 0]], gal) < 5).float().mean(), len(g))
print("(a') new concepts, 3 images per concept, gallery 1.5k + 9.4k:", sc2(Yc, sc_, ic_, keep, mapc, torch.cat([gallery(HC)[torch.as_tensor(pickc).to(DEV)], extra])))
heldc = HELD // 10; hc_set = np.unique(HC // 10); inh = np.isin(heldc, hc_set)  # held-out images (image-holdout) that belong to the 500 concepts
mapi = np.full(5000, -1); mapi[inh] = np.arange(inh.sum()); keep = mapi[ii_] >= 0
print("(c') trained concepts, same 500 concepts (%d images), gallery + 9.4k:" % inh.sum(), sc2(Yi, si_, ii_, keep, mapi, torch.cat([gallery(HELD)[torch.as_tensor(inh).to(DEV)], extra])))
# test images through both models
X = np.concatenate([np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r')[:, 0].reshape(-1, 32, 120) for s in FOLDS[1]])
subj = np.repeat(FOLDS[1], 16000); img = np.tile(np.repeat(np.arange(200), 80), 5)
o = np.random.default_rng(11).permutation(len(X)); subj, img = subj[o], img[o]
T = nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/image_test.npy').reshape(200, 1536)).float()).to(DEV)
yd = torch.as_tensor(np.load(f'{OUT}/test_f1_reve_d6_ho5k_long_optx_things_ep10.npy')).float()
yb = load('reve_d6_hoc500_long_optx_things', 1, 'checkpoint_ep10.pth')(X[o])
for n, y in (('(d) test images, image-holdout model', yd), ('(b) test images, concept-holdout model', yb)):
    y = scaled(y); t = torch.as_tensor(img); gal = torch.cat([T, gallery(HELD)])
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g = torch.as_tensor(np.array([v[j:j + 3] for v in groups.values() for j in range(0, 78, 3)]))
    print(n + ', gallery 200 + 5k: %.3f / %.3f' % (100 * (rank(y, t, gal) < 5).float().mean(), 100 * (rank(y[g].mean(1), t[g[:, 0]], gal) < 5).float().mean()))
