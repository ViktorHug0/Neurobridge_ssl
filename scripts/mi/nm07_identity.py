"""M07 where does a token member hold WHO the trial comes from, where do new people fall, and does moving them help?
M01/M02: a trained person's edge on never-trained images is a person-specific decoding map (late feed-forward layers).
A trial carries no person id (unknown id at test), so the member must infer the person from the trial itself.
(1) identity subspace after each block: span of the 15 Alljoined training people's centroids of the token mean (256-d),
    fitted on trained images; nearest-centroid accuracy on other trials of those people; for NEW people: how far their
    trials fall from the nearest known person (in units of the known people's own spread) and whether trials that fall
    closer to a known person are decoded better.
(2) patches after block L (every token shifted so the token mean's identity coordinates move to a chosen point c):
    neutral   c = centre of the known people (identity removed)
    nearest   c = the nearest known person's centroid (the trial is re-labelled as that person)
    person p  c = one fixed known person (the 3 best-decoded ones)
    half      half-way to the centre
    Never-trained images (first 2,500 of HELD), 5k gallery, r=3 top-5, new vs seen people, paired against the intact member.
    nm07_identity.py <recipe> <fold> <checkpoint>"""
import pandas as pd
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = load(recipe, k, ck); m = f.model; L = len(m.blocks); gal = gallery()
NEW = FOLDS[k]; KNOWN = [s for s in range(1, 21) if s not in NEW]; SEENP = FOLDS[k % 4 + 1]
grab = {}
hooks = [blk.register_forward_hook(lambda mod, i, o, b=b: grab.setdefault(b, []).append(o.mean(1).float().cpu())) for b, blk in enumerate(m.blocks)]


def acts(X):
    grab.clear(); y = f(X, bs=1000)
    return y, {b: torch.cat(v) for b, v in grab.items()}


Xf, sf, _ = cell(KNOWN, SEEN[:240]); half = np.arange(len(sf)) % 2 == 0  # fit on one half of the trials, check on the other
_, Af = acts(Xf); sf = torch.as_tensor(sf)
cells = {'new': cell(NEW, HELD[:2500]), 'seen': cell(SEENP, HELD[:2500])}
base, A = {}, {}
for pn, (X, subj, img) in cells.items():
    y, A[pn] = acts(X); sc, h = score(y, subj, img, gal, hits=True); base[pn] = h['r3']
    q = (nz(scaled(y)) * gal[torch.as_tensor(img)].cpu()).sum(1); cells[pn] = (X, subj, img, q)
    print(pn, 'intact', {a: round(b, 3) for a, b in sc.items()}, flush=True)
for h in hooks:
    h.remove()
SUB = {}
print('(1) identity after each block')
for b in range(L):
    a = Af[b]; cen = torch.stack([a[half & (sf == p).numpy()].mean(0) for p in KNOWN]); mu = cen.mean(0)
    U = torch.linalg.svd(cen - mu, full_matrices=False)[2][:len(KNOWN) - 1].T  # (256, 14) orthonormal basis of the identity subspace
    co = lambda x: (x - mu) @ U
    d = torch.cdist(co(a[~half]), co(cen)); acc = (torch.as_tensor(KNOWN)[d.argmin(1)] == sf[~half]).float().mean().item()
    own = d.gather(1, torch.as_tensor([KNOWN.index(int(p)) for p in sf[~half]])[:, None])[:, 0]  # a known person's trials to their own centroid
    line = f'  block {b}: share of token-mean variance in the subspace {co(a).var(0).sum() / a.var(0).sum():.2f} | known people identified {100 * acc:.1f} % (chance {100 / len(KNOWN):.1f}) | distance to own centroid {own.median():.2f}'
    for pn in ('seen', 'new'):
        an = A[pn][b]; dn = torch.cdist(co(an), co(cen)); near = dn.min(1).values; q = cells[pn][3]
        line += f' | {pn}: to nearest known {near.median():.2f}, r(distance, quality) {np.corrcoef(near, q)[0, 1]:+.3f}'
        if pn == 'new':
            cnt = np.bincount(dn.argmin(1).numpy(), minlength=len(KNOWN)); line += f', most-chosen known person {KNOWN[cnt.argmax()]} ({100 * cnt.max() / cnt.sum():.0f} % of trials)'
    print(line, flush=True); SUB[b] = (U.to(DEV), mu.to(DEV), cen.to(DEV))
# the three best-decoded known people among the seen ones (quality on never-trained images)
qs = {p: cells['seen'][3][cells['seen'][1] == p].mean().item() for p in SEENP}; best = sorted(qs, key=qs.get)[-3:]
print('(2) patches; fixed persons =', best, flush=True)
rows = []


def run(name, b, target):
    U, mu, cen = SUB[b]

    def hook(mod, i, o):
        hm = o.mean(1).float(); c = (hm - mu) @ U  # (B, 14) identity coordinates
        return o + ((target(c, (cen - mu) @ U) - c) @ U.T)[:, None].to(o.dtype)
    hd = m.blocks[b].register_forward_hook(hook); r = dict(patch=name, block=b)
    for pn, (X, subj, img, _) in cells.items():
        sc, h = score(f(X, bs=1000), subj, img, gal, hits=True)
        r[pn] = sc['r3']; r[pn + ' vs intact'] = '%+.1f +- %.1f' % paired(h['r3'], base[pn])
    hd.remove(); rows.append(r); print(r, flush=True)


for b in range(L):
    run('neutral', b, lambda c, cc: torch.zeros_like(c))
    run('half', b, lambda c, cc: c / 2)
    run('nearest', b, lambda c, cc: cc[torch.cdist(c, cc).argmin(1)])
    for p in best:
        run(f'person {p}', b, lambda c, cc, p=p: cc[KNOWN.index(p)].expand_as(c))
pd.set_option('display.width', 250); df = pd.DataFrame(rows).round(3); print(df.to_string(index=False))
df.to_csv(f'{OUT}/m07_identity_{recipe}_f{k}.csv', index=False)
