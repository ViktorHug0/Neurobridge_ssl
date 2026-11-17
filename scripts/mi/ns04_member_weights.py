"""S04 member weights per cohort. The submission sums the six members after scaling each to unit mean norm per batch, so a
member's weight is the same for every cohort. Does a cohort recorded elsewhere want other weights, and does a label-free
batch statistic (the member's mean output norm against its usual value on new Alljoined people) say which?
Datasets: held-out people of fold 1 (reference), ds004018, ds007964 (fold-1 models, known shift applied). r=3.
Per dataset: each member alone, the uniform sum, leave-one-member-out, weights fitted with labels on half of the groups
(coordinate search, scored on the other half), and weights (norm / reference norm)^b per batch.
    ns04_member_weights.py"""
from nc_lib import *

M = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
     ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
short = ['d6', 'rmix', 'sqfx', 'scr', 'sqf', 'labram']; data = {}
X, subj, img = cell(FOLDS[1]); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
data['fold1'] = (torch.as_tensor(X), img, gallery(), torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])))
for c in ('ds004018', 'ds007964'):
    X, subj, img, gal, g3 = cohort(c); data[c] = (X, img, gal, g3)
for name, (X, img, gal, g3) in data.items():
    p = f'{OUT}/s04_{name}.npy'; n = len(X) // 64 * 64
    if not os.path.exists(p):
        Y = np.zeros((len(M), n, 1536), np.float16)
        for j, (r, c) in enumerate(M):
            f = load(r, 1, c); Y[j] = f(X[:n], bs=1000).numpy().astype(np.float16); del f; torch.cuda.empty_cache()
        np.save(p, Y)
for name, (X, img, gal, g3) in data.items():
    Y = torch.as_tensor(np.load(f'{OUT}/s04_{name}.npy').astype(np.float32)); n = Y.shape[1]; g3 = g3[(g3 < n).all(1)]; tg = torch.as_tensor(img)[g3[:, 0]]
    N = Y.norm(dim=2).view(len(M), -1, 64).mean(2)  # (member, batch) mean norm
    if name == 'fold1':
        ref = N.mean(1)
    Z = Y / N.repeat_interleave(64, 1)[:, :, None]  # each member at unit mean norm per batch (= scaled)
    rk = lambda w: rank((w.view(-1, 1, 1) * Z).sum(0)[g3].mean(1) if w.dim() == 1 else (w.repeat_interleave(64, 1)[:, :, None] * Z).sum(0)[g3].mean(1), tg, gal).numpy()
    one = torch.ones(len(M)); b = rk(one); G1 = len(gal) - 1.; half = np.arange(len(b)) % 2; bi = np.random.default_rng(0).integers(0, len(b), (500, len(b)))
    def show(tag, r, sel=slice(None)):
        out = []
        for fn in (lambda v: v < 5, lambda v: v / G1):
            a, c = fn(r[sel]).astype(float), fn(b[sel]).astype(float); i = bi[:, :len(a)] % len(a); out.append('%.3f (%+5.1f +- %.1f %%)' % (a.mean() * (100 if fn(np.array([0]))[0] == 1 else 1), 100 * (a.mean() / c.mean() - 1), 100 * (a[i].mean(1) / c[i].mean(1) - 1).std()))
        print('  %-44s top-5 %s | ranking error %s' % (tag, *out), flush=True)
    print(name, '| mean norm / reference per member: ' + ' '.join('%s %.2f' % (s, v) for s, v in zip(short, (N.mean(1) / ref).tolist())))
    show('uniform sum', b)
    for j, s in enumerate(short):
        show(s + ' alone', rk(torch.eye(len(M))[j]))
    for j, s in enumerate(short):
        show('without ' + s, rk(one - torch.eye(len(M))[j]))
    for h in (0, 1):  # weights fitted with labels on half h (ranking error), scored on the other half
        w = one.clone(); best = np.log1p(rk(w)[half == h]).mean()
        for _ in range(2):
            for j in range(len(M)):
                for v in (0., .5, 1.5, 2., 3.):
                    w2 = w.clone(); w2[j] = v; s_ = np.log1p(rk(w2)[half == h]).mean()
                    if s_ < best:
                        best, w = s_, w2
        show('weights fitted on the other half ' + ' '.join('%.1f' % v for v in w.tolist()), rk(w), half != h)
    for be in (1., 2., 4.):
        show('weights (batch norm / reference)^%d' % be, rk((N / ref[:, None]) ** be))
