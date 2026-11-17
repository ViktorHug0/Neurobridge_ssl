"""R06b the label-free confidence estimator on cohorts whose offset is known from labelled sweeps (nq07: ds007964 +3..+6,
ds004018 +4..+6 samples; THINGS-EEG2 +5 when its lag is not corrected). Frozen member, batches of 64 mixing the cohort's
people, input shifted by c (content later by c): mean output norm per batch -> argmax over c.
    nr06_autolag_cohorts.py <recipe> <fold> <checkpoint>"""
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]; f = load(recipe, k, ck); tt = torch.arange(120); C = list(range(-4, 12))
for name, files in (('ds007964', sorted(glob.glob('/nasbrain/p20fores/ds007964/export/sub-*.npz'))[:6]), ('ds004018', sorted(glob.glob('/nasbrain/p20fores/ds004018/export/sub-*.npz'))[:6])):
    X = torch.cat([torch.as_tensor(np.load(p)['X'][:3000].astype(np.float32)) for p in files]); X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(0))]; nb = len(X) // 64
    N = np.stack([f(X[:, :, (tt - c).clamp(0, 119)], bs=1000).cpu().float().norm(dim=1)[:nb * 64].view(nb, 64).mean(1).numpy() for c in C], 1); est = np.array(C)[N.argmax(1)]
    print('%s (%d people, %d batches): mean norm by correction ' % (name, len(files), nb) + ' '.join('%+d:%.1f' % (c, v) for c, v in zip(C, N.mean(0))) + ' | per-batch estimate mean %+.2f sd %.2f' % (est.mean(), est.std()), flush=True)
