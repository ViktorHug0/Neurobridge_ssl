"""P01 the resolution of the decoded code: are the misses near the target (the neighbourhood is right, the fine
detail is missing) or anywhere (noise)?  Cached outputs of the lag five (+ LaBraM), new people, never-trained images,
5k gallery, r = 1 / 3 / 4 and "all" (every trial of the image over the fold's 5 people, ~20).
(a) pairwise discrimination P(target scores above image j) for j = the target's k-th nearest gallery neighbour
    (k = 1 .. 2500) and for a random image (chance .5): a flat curve = noise-limited, a drop at small k = resolution-limited;
(b) concept level (THINGS concept = image index // 10; a held image has ~2 siblings in the gallery): top-5 holds an
    image of the target's concept, against the image top-5; P(target above a sibling);
(c) the wrong winner (top-1 when it is not the target): its rank among the target's neighbours, and hubs (share of
    the queries won by the 1 % most-retrieved gallery images; uniform = 1 %).
    np01_resolution.py <folds> [six]"""
from nc_lib import *
from nc06_names import NAMES

folds = [int(a) for a in sys.argv[1].split(',')]; SIX = 'six' in sys.argv
Y = lambda k, r, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')).float())
G = gallery(); GS = G @ G.T; GS.fill_diagonal_(2); order = GS.argsort(1, descending=True)[:, 1:]  # (5000, 4999) neighbours, nearest first
pos = torch.empty(5000, 5000, dtype=torch.int16, device=DEV); pos.scatter_(1, order, torch.arange(4999, device=DEV, dtype=torch.int16).expand(5000, -1))  # pos[t, j] = neighbour rank of j for t
conc = torch.as_tensor(HELD // 10).to(DEV); KS = [1, 2, 5, 10, 20, 50, 100, 500, 2500]
sib = (conc[:, None] == conc[None]) & ~torch.eye(5000, dtype=torch.bool, device=DEV)
print('gallery: cosine to the k-th neighbour', ' '.join('k%d %.3f' % (k, GS.gather(1, order[:, k - 1:k]).mean()) for k in KS), '| siblings per image %.2f, nearest neighbour is a sibling %.2f' % (sib.sum(1).float().mean(), sib.gather(1, order[:, :1]).float().mean()))
acc = {}
for k in folds:
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    y = sum([Y(k, 'd6', 12), Y(k, 'rmix', 10), Y(k, 'sqfx', 40)] + ([Y(k, 'scr', 12), Y(k, 'sqf', 32)] if k in (1, 2) else [scaled(torch.as_tensor(np.load(f'{R}/logs/holdout/f{k}/{m}.npy')).float()) for m in NAMES[k][3:]]))  # fold 3: REVE-scratch and SQF are the cached non-lag ones, as in ni05
    if SIX:
        y = y + scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_plab_ho5k_lag5_long_ep10_raw.npy')).float())
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); t4 = img[g4[:, 0]]
    n = torch.bincount(img, minlength=5000).float()
    Q = {'r1': (y, img), 'r3': (y[g4[:, :3]].mean(1), t4), 'r4': (y[g4].mean(1), t4), 'all': ((torch.zeros(5000, 1536).index_add_(0, img, y) / n[:, None])[n > 0], torch.nonzero(n > 0)[:, 0])}
    gen = torch.Generator(device=DEV).manual_seed(0)
    for name, (q, t) in Q.items():
        a = acc.setdefault(name, dict(n=0, pair=torch.zeros(len(KS) + 1), img5=0, con5=0, sib=0., sibn=0, wpos=[], win=torch.zeros(5000)))
        for b in range(0, len(q), 4096):
            S = nz(q[b:b + 4096]).to(DEV) @ G.T; tt = t[b:b + 4096].to(DEV); tg = S.gather(1, tt[:, None])
            cols = torch.cat([order[tt][:, [kk - 1 for kk in KS]], torch.randint(0, 5000, (len(tt), 1), device=DEV, generator=gen)], 1)
            a['pair'] += (tg > S.gather(1, cols)).float().sum(0).cpu(); a['n'] += len(tt)
            top = S.topk(5).indices; a['img5'] += (top == tt[:, None]).any(1).sum().item(); a['con5'] += (conc[top] == conc[tt][:, None]).any(1).sum().item()
            m = sib[tt]; a['sib'] += ((S < tg) & m).sum().item(); a['sibn'] += m.sum().item()
            w = torch.where(top[:, 0] == tt, top[:, 1], top[:, 0]); a['wpos'].append(pos[tt, w].cpu()); a['win'] += torch.bincount(w, minlength=5000).float().cpu()
for name, a in acc.items():
    wp = torch.cat(a['wpos']).float(); hub = a['win'].sort(descending=True).values
    print('%-3s n %6d | P(target > k-th neighbour): %s | random %.3f' % (name, a['n'], ' '.join('k%d %.3f' % (k, v) for k, v in zip(KS, a['pair'][:-1] / a['n'])), a['pair'][-1] / a['n']))
    print('    image top-5 %.3f %%, concept top-5 %.2f %% (chance ~%.2f) | P(target > sibling) %.3f | wrong winner: among the target\'s 50 nearest %.1f %% (chance 1.0), 500 nearest %.1f %% (chance 10), median neighbour rank %d | hubs: top 1 %% of gallery images win %.1f %% of the queries, top 10 %% win %.1f %%'
          % (100 * a['img5'] / a['n'], 100 * a['con5'] / a['n'], 100 * 5 * (1 + sib.sum(1).float().mean().item()) / 5000, a['sib'] / a['sibn'], 100 * (wp < 50).float().mean(), 100 * (wp < 500).float().mean(), wp.median(), 100 * hub[:50].sum() / hub.sum(), 100 * hub[:500].sum() / hub.sum()), flush=True)
