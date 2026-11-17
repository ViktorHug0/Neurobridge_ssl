"""R04b are the official test-run trials harder because of the RUNS or because of the IMAGES? (E11 left the cause open;
the sealed set is also about twice as hard as the proxy predicts.) Frozen rmix lag f1 @10, held-out people, single trials,
ONE common gallery (200 test images + the 5k held-out images): per image, the mean single-trial P(target > random image).
Hits depend strongly on image typicality (P02/P03), so test images are compared with held-out training-run images of the
same typicality (cosine to the gallery mean), per quintile of the common typicality distribution.
    nr04_testrun_vs_image.py"""
from nc_lib import *

k = 1; D = f'{R}/data/things_eeg/image_feature/dinov2-giant_track1'
T = nz(torch.tensor(np.load(f'{D}/image_test.npy').reshape(200, 1536)).float()).to(DEV); gal = torch.cat([T, gallery()])
typ = (gal @ nz(gal.mean(0, keepdim=True)).T)[:, 0].cpu().numpy()
# training-run trials of the held-out people on the 5k held-out images (cached stream) and their test-run trials (ne09 cache)
st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); ya = scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_rmix_ho5k_lag5_ep10_raw.npy')).float()); ia = torch.as_tensor(st['img']) + 200; sa = st['subj']
img = np.tile(np.repeat(np.arange(200), 80), 5); subj = np.repeat(FOLDS[k], 16000); o = np.random.default_rng(11).permutation(len(img))  # = ne09's order
yb = scaled(torch.as_tensor(np.load(f'{OUT}/test_f{k}_mi_rmix_ho5k_lag5_ep10.npy').astype(np.float32))); ib = torch.as_tensor(img[o]); sb = subj[o]
auc = lambda y, t: 1 - rank(y, t, gal).float().cpu().numpy() / (len(gal) - 1)
A, B = auc(ya, ia), auc(yb, ib)
print('single-trial P(target > random), common %d gallery: training-run trials %.4f | test-run trials %.4f' % (len(gal), A.mean(), B.mean()))
print('per person (train-run / test-run): ' + ' | '.join('%d: %.3f / %.3f' % (p, A[sa == p].mean(), B[sb == p].mean()) for p in FOLDS[k]))
edges = np.quantile(typ, [.2, .4, .6, .8]); qa, qb = np.searchsorted(edges, typ[ia.numpy()]), np.searchsorted(edges, typ[ib.numpy()])
print('typicality (cosine to gallery mean): held-out images %.3f +- %.3f | test images %.3f +- %.3f' % (typ[200:].mean(), typ[200:].std(), typ[:200].mean(), typ[:200].std()))
print('by typicality quintile (atypical -> typical), train-run / test-run (share of test images): ' + ' | '.join('%.3f / %.3f (%.0f %%)' % (A[qa == i].mean(), B[qb == i].mean() if (qb == i).any() else float('nan'), 100 * (np.searchsorted(edges, typ[:200]) == i).mean()) for i in range(5)))
w = np.bincount(qb, minlength=5) / len(qb); print('training-run trials re-weighted to the test images typicality mix: %.4f (test-run %.4f)' % (sum(w[i] * A[qa == i].mean() for i in range(5)), B.mean()))
# top-5 at r=1 for reference
print('top-5 %% r=1: training-run %.3f | test-run %.3f' % (100 * (rank(ya, ia, gal) < 5).float().mean(), 100 * (rank(yb, ib, gal) < 5).float().mean()))
# where in the ranking does the test-run deficit sit, and is it the typicality mix?
ra, rb = rank(ya, ia, gal).cpu().numpy(), rank(yb, ib, gal).cpu().numpy()
print('share of trials with the target in the top n (training-run / test-run): ' + ' | '.join('top-%d %.3f / %.3f %%' % (n, 100 * (ra < n).mean(), 100 * (rb < n).mean()) for n in (5, 20, 100, 500, 2600)))
print('top-5 %% by typicality quintile (atypical -> typical), training-run / test-run: ' + ' | '.join('%.2f / %.2f' % (100 * (ra[qa == i] < 5).mean(), 100 * (rb[qb == i] < 5).mean()) for i in range(5)))
print('training-run top-5 re-weighted to the test typicality mix: %.3f %% (test-run %.3f %%)' % (100 * sum(w[i] * (ra[qa == i] < 5).mean() for i in range(5)), 100 * (rb < 5).mean()))
# how often is each kind of image the winner (top-1) of a query of the OTHER kind? hubs
S = lambda y: (nz(y).to(DEV) @ gal.T).argmax(1).cpu().numpy()
wa, wb = S(ya), S(yb); print('winner (top-1) is a test image: for training-run queries %.2f %%, for test-run queries %.2f %% (test images are %.2f %% of the gallery)' % (100 * (wa < 200).mean(), 100 * (wb < 200).mean(), 100 * 200 / len(gal)))
# per image: mean single-trial top-5 rate; how concentrated are the hits over images
ha = np.bincount(ia.numpy() - 200, ra < 5, 5000) / np.maximum(np.bincount(ia.numpy() - 200, minlength=5000), 1); hb = np.bincount(ib.numpy(), rb < 5, 200) / 400
print('images with at least one top-5 hit: training-run %.1f %% of 5,000 (20 trials each) | test %.1f %% of 200 (400 trials each); hits held by the best 10 %% of images: %.0f %% / %.0f %%' % (100 * (ha > 0).mean(), 100 * (hb > 0).mean(), 100 * np.sort(ha)[-500:].sum() / ha.sum(), 100 * np.sort(hb)[-20:].sum() / hb.sum()))
