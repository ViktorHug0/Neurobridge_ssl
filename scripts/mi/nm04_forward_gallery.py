"""M04 why does a gallery made of OTHER PEOPLE's outputs beat the image embeddings for a new person (M01), and can the
grader's fixed gallery be used that way? The population pattern of an image, pbar(j) = mean output of the 15 training
people for image j, is what a new person's output should be compared with. Forward model: pbar(j) ~ G d_j (d_j = image
embedding; ridge, cross-fitted over images: every image is predicted by a map that never saw it).
Galleries, new people x never-trained images, 5k, exact ranks, r=1 / r=3, raw (uncentred) queries:
    d_j            the grader's gallery
    pbar(j)        other people's outputs (needs their EEG for the gallery image: not deployable)
    G d_j          forward-predicted population pattern (cosine; and dot product minus half its squared norm)
    G^T y vs d_j   the same forward model moved to the output side = deployable (the grader's cosine against d_j)
    nm04_forward_gallery.py <recipe> <fold> <checkpoint>"""
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NEW = FOLDS[k]; TRAIN = [s for s in range(1, 21) if s not in NEW]
p = f'{OUT}/m04_out_{recipe}_f{k}.npy'
if not os.path.exists(p):
    f = load(recipe, k, ck)
    X = np.concatenate([np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[HELD]).transpose(1, 0, 2, 3).reshape(20000, 32, 120) for s in TRAIN + NEW])
    np.save(p, f(X, bs=1000).numpy().astype(np.float16)); np.save(p[:-4] + '_ok.npy', np.abs(X).sum((1, 2)) > 0); del X, f; torch.cuda.empty_cache()
O = torch.as_tensor(np.load(p)).float().view(20, 4, 5000, 1536); ok = torch.as_tensor(np.load(p[:-4] + '_ok.npy')).view(20, 4, 5000)
O = O / O.norm(dim=-1).clamp_min(1e-6)[ok].mean()  # one global scale (the per-batch scaling of the proxy)
w = ok[..., None].float()
pbar = ((O[:15] * w[:15]).sum(1) / w[:15].sum(1).clamp(min=1)).mean(0)  # (5000, 1536) mean over training people of their repeat means
d = gallery().cpu()
fold = np.random.default_rng(0).integers(0, 5, 5000)


def forward_map(lam):  # cross-fitted ridge d -> pbar; returns the predicted gallery (5000, 1536) and one G per image fold
    P, G = torch.zeros_like(pbar), []
    for v in range(5):
        tr, te = torch.as_tensor(fold != v), torch.as_tensor(fold == v)
        A = d[tr].double(); g = torch.linalg.solve(A.T @ A + lam * torch.eye(1536, dtype=torch.double), A.T @ pbar[tr].double()).float()  # (1536 d, 1536 out)
        P[te] = d[te] @ g; G.append(g)
    return P, G


v4 = ok[15:].all(1)  # (5, 5000) new person has 4 repeats
q1 = O[15:, 0][v4]; q3 = O[15:, :3].mean(1)[v4]; t = torch.cat([torch.nonzero(v)[:, 0] for v in v4]); fq = torch.as_tensor(fold)[t]


def top5(q, gal, bias=None):
    if bias is None:
        return 100 * (rank(q, t, nz(gal).to(DEV)) < 5).float().mean().item()
    S = q.to(DEV) @ gal.to(DEV).T + bias.to(DEV)  # dot-product score with a per-image offset
    return 100 * ((S > S.gather(1, t.to(DEV)[:, None])).sum(1) < 5).float().mean().item()


show = lambda n, f1: print('%-58s r1 %.3f   r3 %.3f' % (n, f1(q1), f1(q3)), flush=True)
show('image embeddings d_j (the grader)', lambda q: top5(q, d))
show('population patterns pbar(j), cosine', lambda q: top5(q, pbar))
show('population patterns, centred on their mean, cosine', lambda q: top5(q, pbar - pbar.mean(0)))
show('population patterns, dot - |pbar|^2 / 2', lambda q: top5(q, pbar, -.5 * pbar.pow(2).sum(1)))
for lam in (.01, .1, 1., 10.):
    P, G = forward_map(lam)
    r2 = 1 - (P - pbar).pow(2).sum() / (pbar - pbar.mean(0)).pow(2).sum()
    print(f'-- forward model, ridge {lam}: cross-fitted R2 of pbar {r2:.3f}; cosine(G d_j, pbar(j)) {torch.cosine_similarity(P, pbar).mean():.3f} vs cosine(d_j, pbar(j)) {torch.cosine_similarity(d, pbar).mean():.3f}')
    show(f'   G d_j, cosine', lambda q: top5(q, P))
    show(f'   G d_j, dot - |G d_j|^2 / 2', lambda q: top5(q, P, -.5 * P.pow(2).sum(1)))
    back = lambda q: torch.cat([q[fq == v] @ G[v].T for v in range(5)])  # y -> G^T-side: (n, 1536 d); regroup targets accordingly
    tt = torch.cat([t[fq == v] for v in range(5)])
    show(f'   G^T y against d_j (deployable form)', lambda q: 100 * (rank(back(q), tt, d.to(DEV)) < 5).float().mean().item())
