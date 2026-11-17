"""U03 a correction of the ensemble output fitted on images the members never saw (stacking across IMAGES).
M04: for a never-trained image the members' expected output m_j differs from its embedding d_j in an image-specific,
person-independent way, and ranking against m_j instead of d_j gives new people +16-18 %. The gallery is fixed, so the
correction has to be a map g on the output: g(y) closer to d_j than y is. M04 / M05 tried linear maps fitted on new
PEOPLE's outputs; here g is fitted on never-trained IMAGES, with the trials of the fold's 15 Alljoined TRAINING people
(strong outputs, 60 trials per image), and tested on other never-trained images seen by the held-out people.
Fold 1, sub16 composition. The 5,000 never-trained images are split in halves A (fit) / B (test, gallery = B).
g: residual linear map (zero-init, weight decay) or residual MLP; loss = cross-entropy over the A gallery (cosine / 0.05)
of the corrected single-trial output. Test: held-out people x B, r=3 (g per trial, then the grader's average).
    nu03_image_bias.py"""
from nc_lib import *
from torch import nn

k = 1; M = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
            ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
train = [s for s in range(1, 21) if s not in FOLDS[k]]; D = nz(torch.tensor(np.asarray(TARGETS[HELD])).float()).to(DEV); torch.manual_seed(0)
p = f'{OUT}/u03_f{k}.npz'
if not os.path.exists(p):
    S = {}
    for tag, people in (('new', FOLDS[k]), ('seen', train)):
        X, subj, img = cell(people); X = torch.as_tensor(X); y = 0
        for r, c in M:
            f = load(r, k, c); y = y + scaled(f(X, bs=1000)); del f; torch.cuda.empty_cache()
        S[tag + '_y'], S[tag + '_subj'], S[tag + '_img'] = y.numpy().astype(np.float16), subj, img; print('cached', tag, len(img), flush=True)
    np.savez(p, **S)
S = np.load(p); A = np.arange(5000) % 2 == 0  # image halves in HELD order
ys, is_ = torch.as_tensor(S['seen_y'].astype(np.float32)), torch.as_tensor(S['seen_img']); yn, in_, sn = torch.as_tensor(S['new_y'].astype(np.float32)), S['new_img'], S['new_subj']
fit = torch.as_tensor(A[is_.numpy()]); Xf, Tf = ys[fit], is_[fit]; Xv, Tv = ys[~fit], is_[~fit]  # seen people: A to fit, B to watch
groups = {}
for i, kk in enumerate(zip(sn.tolist(), in_.tolist())):
    if not A[kk[1]]:
        groups.setdefault(kk, []).append(i)
g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(in_)[g3[:, 0]]; galB = D[~torch.as_tensor(A)]; posB = torch.cumsum(~torch.as_tensor(A), 0) - 1


def test(g):  # held-out people x B images, r=3, gallery B
    with torch.no_grad():
        y = torch.cat([g(yn[b:b + 8192].to(DEV)).cpu() for b in range(0, len(yn), 8192)])
    return rank(y[g3].mean(1), posB[tg], galB).numpy()


base = test(lambda y: y); bi = np.random.default_rng(0).integers(0, len(base), (1000, len(base)))


def versus(r):
    o = []
    for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 2499.):
        a, c = fn(r).astype(float), fn(base).astype(float); o.append('%+5.1f +- %.1f' % (100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
    return 'top-5 %.3f %% (%s) | top-100 %s | ranking error %s' % (100 * (r < 5).mean(), *o)


print('fold 1, new people x %d never-trained images B (gallery 2,500), r=3: no correction top-5 %.3f %%, top-100 %.2f %%, ranking error %.4f | fit: %d trials of the 15 training people on the other 2,500 images' % (
    len(base), 100 * (base < 5).mean(), 100 * (base < 100).mean(), (base / 2499.).mean(), len(Xf)))
# oracle of M04 on this split: rank against the training people's mean output for each B image instead of its embedding
mB = nz(torch.zeros(5000, 1536).index_add_(0, is_, nz(ys)))[~torch.as_tensor(A)].to(DEV)
print('  oracle gallery = mean output of the training people for the B images (M04):', versus(rank(yn[g3].mean(1), posB[tg], mB).numpy()))
galA = D[torch.as_tensor(A)]; posA = (torch.cumsum(torch.as_tensor(A), 0) - 1).to(DEV)
for name, make, wd in (('linear residual, weight decay 1e-2', lambda: nn.Linear(1536, 1536, bias=False), 1e-2), ('linear residual, weight decay 1', lambda: nn.Linear(1536, 1536, bias=False), 1.),
                       ('rank-64 residual, weight decay 1e-2', lambda: nn.Sequential(nn.Linear(1536, 64, bias=False), nn.Linear(64, 1536, bias=False)), 1e-2),
                       ('MLP residual 1024, weight decay 1e-2', lambda: nn.Sequential(nn.Linear(1536, 1024), nn.GELU(), nn.Linear(1024, 1536)), 1e-2),
                       ('MLP residual 1024, weight decay 1', lambda: nn.Sequential(nn.Linear(1536, 1024), nn.GELU(), nn.Linear(1024, 1536)), 1.)):
    h = make().to(DEV); last = h if isinstance(h, nn.Linear) else h[-1]; nn.init.zeros_(last.weight); g = lambda y, h=h: y + y.norm(dim=1, keepdim=True) * h(nz(y))
    opt = torch.optim.AdamW(h.parameters(), 3e-4, weight_decay=wd)
    for ep in range(6):
        h.train()
        for b in torch.randperm(len(Xf)).split(1024):
            loss = nn.functional.cross_entropy(nz(g(Xf[b].to(DEV))) @ galA.T / 0.05, posA[Tf[b].to(DEV)]); opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():  # watch: the training people on the B images, single trials, gallery B
            yv = torch.cat([g(Xv[b:b + 8192].to(DEV)).cpu() for b in range(0, len(Xv), 8192)]); rv = rank(yv[::7], posB[Tv[::7]], galB).numpy(); r0 = rank(Xv[::7], posB[Tv[::7]], galB).numpy()
        print('  %-36s epoch %d fit loss %.3f | training people on B, r=1 top-5 %.2f %% (none %.2f) | NEW people on B: %s' % (name, ep + 1, loss.item(), 100 * (rv < 5).mean(), 100 * (r0 < 5).mean(), versus(test(g))), flush=True)
