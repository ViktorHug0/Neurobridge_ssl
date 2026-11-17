"""R01 which simple nuisance separates a weak new person from a strong one? (frozen member, held-out people, never-trained
images, 5k gallery, r=3). Per held-out person, the input is (a) shifted in time by s samples (content later by s), or
(b) scaled by g. Per person and transform: top-5 %, top-100 % and 1 - AUC (mean rank / 4999; lower = better).
If a weak person's optimum sits away from (0, 1), part of their deficit is a latency or a gain mismatch that a stateless
front-end could remove; if every person peaks at (0, 1), these two nuisances are not the cause.
    nr01_person_probe.py <recipe> <fold> <checkpoint>"""
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]; f = load(recipe, k, ck); gal = gallery(); X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); t = torch.arange(120)
groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]; who = subj[g4[:, 0].numpy()]


def row(name, Y):
    r = rank(scaled(f(Y, bs=1000).cpu())[g4[:, :3]].mean(1), tg, gal).cpu().numpy()
    print('%-10s' % name + ' | '.join('%2d: %.2f %5.2f %.4f' % (p, 100 * (r[who == p] < 5).mean(), 100 * (r[who == p] < 100).mean(), r[who == p].mean() / 4999) for p in FOLDS[k]) + ' | all: %.2f %5.2f %.4f' % (100 * (r < 5).mean(), 100 * (r < 100).mean(), r.mean() / 4999), flush=True)


print(recipe, ck, 'fold', k, '| per person: top-5 %, top-100 %, 1 - AUC')
for s in range(-4, 5):
    row('shift %+d' % s, X[:, :, (t - s).clamp(0, 119)])
for g in (0.5, 0.7, 1.4, 2.0):
    row('gain %.1f' % g, X * g)
