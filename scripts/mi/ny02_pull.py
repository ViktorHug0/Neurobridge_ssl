"""Y02 is the output for a NEW image pulled toward the TRAINED images that resemble it, and does undoing the pull help? (M04: other
people's outputs are a better gallery than the image embeddings by +16..18 %: an image-specific, person-independent distortion.)
Cache of M04: rmix lag fold 1, 20 people x 4 repeats x the 5,000 never-trained images. pbar(j) = mean output of the 15 training people.
(1) what pbar(j) is made of: cosine with its own embedding d_j, with the mean embedding of j's k nearest TRAINED images, centred on the mean target;
(2) deployable repulsion: y' = y - g * |y| * unit(sum_k softmax(b * cos(y, t_k)) t_k) over the 11,540 trained targets (new people, r=3, 5k gallery).
    ny02_pull.py"""
from nc_lib import *

O = torch.as_tensor(np.load(f'{OUT}/m04_out_mi_rmix_ho5k_lag5_f1.npy')).float().view(20, 4, 5000, 1536); ok = torch.as_tensor(np.load(f'{OUT}/m04_out_mi_rmix_ho5k_lag5_f1_ok.npy')).view(20, 4, 5000)
O = O / O.norm(dim=-1).clamp_min(1e-6)[ok].mean(); w = ok[..., None].float()
pbar = ((O[:15] * w[:15]).sum(1) / w[:15].sum(1).clamp(min=1)).mean(0)
T = torch.tensor(np.asarray(TARGETS)).float(); tr = np.setdiff1d(np.arange(16540), HELD); d, B = nz(T[HELD]), nz(T[tr]); mu = B.mean(0)
S = d @ B.T; c = lambda a, b: torch.cosine_similarity(a - (a @ nz(mu[None])[0])[:, None] * nz(mu[None])[0], b - (b @ nz(mu[None])[0])[:, None] * nz(mu[None])[0]).mean().item()  # cosine after removing the mean-target direction
print('pbar(j) vs its own embedding d_j: %.3f' % c(pbar, d))
for k in (1, 5, 20, 100):
    nb = nz(B[S.topk(k, 1).indices].mean(1)); print('pbar(j) vs mean of the %3d nearest trained images: %.3f   (d_j vs the same: %.3f)' % (k, c(pbar, nb), c(d, nb)))
# partial: does pbar carry the part of d_j that the trained neighbours do NOT explain?
nb = nz(B[S.topk(20, 1).indices].mean(1)); res = d - (d * nb).sum(1, keepdim=True) * nb
print('pbar(j) vs the part of d_j orthogonal to its 20-neighbour mean: %.3f   | vs the neighbour mean: %.3f' % (c(pbar, res), c(pbar, nb)))
v4 = ok[15:].all(1); q3 = O[15:, :3].mean(1)[v4]; t = torch.cat([torch.nonzero(v)[:, 0] for v in v4]); G = d.to(DEV); Bd = B.to(DEV)
top = lambda q: (100 * (rank(q, t, G) < 5).float().mean().item(), 100 * (rank(q, t, G) < 100).float().mean().item())
print('new people, r=3, 5k: intact top-5 %.3f top-100 %.2f' % top(q3))
for b in (20., 50.):
    P = torch.cat([nz(torch.softmax(b * (nz(q.to(DEV)) @ Bd.T), 1) @ Bd).cpu() for q in q3.split(4096)])
    print('kernel width', b, ' cosine(y, its trained-neighbour mean) %.3f' % torch.cosine_similarity(q3, P).mean())
    for g in (-.2, -.1, -.05, .05, .1, .2, .4):
        print('   y - %+.2f |y| P: top-5 %.3f top-100 %.2f' % ((g,) + top(q3 - g * q3.norm(dim=1, keepdim=True) * P)), flush=True)
