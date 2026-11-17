"""M05 a full linear output map fitted for NEW-person outputs (stacking). M01/M04: a new person's output matches other
people's outputs better than the image embeddings, so the head (fitted on seen people) may not be the best linear readout
of what new people produce. E01 closed diagonal re-weightings and ridge; here W (1536 x 1536, identity at start) is
trained on the retrieval loss itself, on out-of-fold outputs: ens5 (cached, scaled sum) of the new people of two folds x
half the never-trained images; tested on the third fold's people x the other half (gallery = that half + 9.4k other
THINGS images, exact ranks, r=3 and r=1), paired against W = identity. Folds rotated and pooled.
    nm05_stack.py [lr, default 3e-5] [rank of the update, default full]"""
import torch.nn.functional as Fn
from nc_lib import *

lr = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-5; rk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
galb = gallery(big=True); half = torch.as_tensor(np.random.default_rng(0).permutation(5000) < 2500)  # image in the fitting half
D = {}
for k in (1, 2, 3):
    F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    y = sum(scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in ENS5[k])
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4]))
    D[k] = dict(y=y, img=img, g3=g3[~half[img[g3[:, 0]]]]); print('fold', k, flush=True)
test_gal = torch.cat([galb[:5000][~half.to(DEV)], galb[5000:]]); pos = torch.cumsum(~half, 0) - 1  # image -> row of the test gallery
fit_gal = galb[:5000][half.to(DEV)]; fpos = torch.cumsum(half, 0) - 1


@torch.no_grad()
def test(k, W):
    d = D[k]; y = d['y'] if W is None else torch.cat([(d['y'][b:b + 8192].to(DEV) @ W.T).cpu() for b in range(0, len(d['y']), 8192)])
    t3 = pos[d['img'][d['g3'][:, 0]]]; one = ~half[d['img']]
    return (rank(y[d['g3']].mean(1), t3, test_gal) < 5).numpy(), (rank(y[one], pos[d['img'][one]], test_gal) < 5).numpy()


base = {k: test(k, None) for k in D}
print('identity: r3 %.3f  r1 %.3f' % tuple(100 * np.concatenate([base[k][i] for k in D]).mean() for i in (0, 1)), flush=True)
H = {}
for k in D:
    tr = [j for j in D if j != k]
    Y = torch.cat([D[j]['y'][half[D[j]['img']]] for j in tr]); T = torch.cat([fpos[D[j]['img'][half[D[j]['img']]]] for j in tr])
    eye = torch.eye(1536, device=DEV)
    P = [torch.zeros(1536, rk, device=DEV).requires_grad_(True), (torch.randn(rk, 1536, device=DEV) * .01).requires_grad_(True)] if rk else [torch.zeros(1536, 1536, device=DEV).requires_grad_(True)]
    Wf = lambda: eye + (P[0] @ P[1] if rk else P[0])
    s = torch.tensor(3.4, device=DEV, requires_grad=True)  # log logit scale
    opt = torch.optim.Adam([dict(params=P, lr=lr), dict(params=[s], lr=1e-2)]); g = torch.Generator().manual_seed(0)
    for ep in range(8):
        perm = torch.randperm(len(Y), generator=g); tot = 0
        for i, b in enumerate(range(0, len(perm) - 2047, 2048)):
            idx = perm[b:b + 2048]
            loss = Fn.cross_entropy(s.exp() * Fn.normalize(Y[idx].to(DEV) @ Wf().T, dim=1) @ fit_gal.T, T[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        h = test(k, Wf().detach()); H.setdefault(ep, {})[k] = h
        print(f'left-out fold {k} epoch {ep + 1}: fit loss {tot / (i + 1):.4f} scale {s.exp().item():.1f} | r3 %.3f (%+.1f +- %.1f %%)  r1 %.3f (%+.1f +- %.1f %%)'
              % (100 * h[0].mean(), *paired(h[0], base[k][0]), 100 * h[1].mean(), *paired(h[1], base[k][1])), flush=True)
for ep in H:
    cat = lambda d, i: np.concatenate([d[k][i] for k in D])
    print(f'pooled, epoch {ep + 1}: r3 %.3f (%+.1f +- %.1f %%)  r1 %.3f (%+.1f +- %.1f %%)' % (100 * cat(H[ep], 0).mean(), *paired(cat(H[ep], 0), cat(base, 0)), 100 * cat(H[ep], 1).mean(), *paired(cat(H[ep], 1), cat(base, 1))))
