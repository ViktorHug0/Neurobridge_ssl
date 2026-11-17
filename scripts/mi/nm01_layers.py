"""M01 where inside a token member does a new person's trial stop matching other people's?
After every block the member's own readout is applied (token mean -> head: a 1536-d "lens" of that depth). In that space
the pattern of a (person, image) is compared with (a) the image embeddings = what the member is trained for, and (b) OTHER
PEOPLE's patterns for the same image, a "neural gallery" with no image embedding involved (patterns minus the person's
own mean pattern). Never-trained images (HELD, 5,000), fold k, cosine, exact top-5 among the 5,000 images.
Query people: seen = five training people of the fold model, new = its five held-out people.
    image   gallery = image embeddings (the real readout at the last block)
    pop10   gallery = mean pattern of the 10 other training people (never the query person)
    seen4   gallery = 4 seen people (the other four for a seen query)      } matched size: is a new person less aligned
    new4    gallery = 4 new people (the other four for a new query)        } with everyone, or only with training people?
    own     gallery = the query person's own other repeats (r=1 query only): the person's private, repeatable code
r=1 = one trial, r=3 = mean of three repeats.
    nm01_layers.py <recipe> <fold> <checkpoint>"""
import pandas as pd
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NEW, SEENP = FOLDS[k], FOLDS[k % 4 + 1]
POP = [s for s in range(1, 21) if s not in NEW + SEENP]
f = load(recipe, k, ck); m = f.model; L = len(m.blocks); grab = {}
for b, blk in enumerate(m.blocks):
    blk.register_forward_hook(lambda mod, i, o, b=b: grab.__setitem__(b, m.head(o.mean(1)).float()))
dino = gallery()


@torch.no_grad()
def acts(s):
    """lens patterns (L, 4, 5000, 1536) float16 on the CPU (repeats first) and which (repeat, image) exist."""
    x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[HELD]).transpose(1, 0, 2, 3)
    ok = torch.as_tensor(np.abs(x).sum((2, 3)) > 0); x = x.reshape(20000, 32, 120); out = []
    for b in range(0, 20000, 2000):
        xb = torch.as_tensor(x[b:b + 2000]).to(DEV)
        run_eeg_backbone(m, f.cfg, xb, torch.full((len(xb),), 10 ** 6, device=DEV))
        out.append(torch.stack([grab[l] for l in range(L)]).half().cpu())
    return torch.cat(out, 1).view(L, 4, 5000, -1), ok


def top5(q, gal, valid):
    return 100 * (rank(q[valid], torch.nonzero(valid)[:, 0], nz(gal.float().to(DEV) + 1e-12)) < 5).float().mean().item()


def centred(a, ok):  # minus the person's mean pattern over its existing trials; missing trials -> 0
    w = ok[..., None].float(); a = a.float()
    return (a - (a * w).sum((0, 1)) / w.sum()) * w


pop, P = [0] * L, {}
for s in POP:
    A, ok = acts(s)
    for l in range(L):
        pop[l] = pop[l] + centred(A[l], ok).sum(0) / ok.sum(0).clamp(min=1)[:, None]
    print('gallery person', s, flush=True)
for s in SEENP + NEW:
    P[s] = acts(s)
rows = []
for l in range(L):
    C = {s: centred(P[s][0][l], P[s][1]) for s in P}
    mean = {s: C[s].sum(0) / P[s][1].sum(0).clamp(min=1)[:, None] for s in P}
    for grp, people in (('seen', SEENP), ('new', NEW)):
        r = dict(after_block=l, query=grp)
        add = lambda key, v: r.__setitem__(key, r.get(key, 0) + v / 5)
        for s in people:
            c, raw, v = C[s], P[s][0][l].float(), P[s][1].all(0)
            others = lambda g: sum(mean[t] for t in [t for t in g if t != s][:4])
            add('image r1', top5(raw[0], dino, v)); add('image r3', top5(raw[:3].mean(0), dino, v))
            for gn, gal in (('pop10', pop[l]), ('seen4', others(SEENP)), ('new4', others(NEW))):
                add(f'{gn} r1', top5(c[0], gal, v)); add(f'{gn} r3', top5(c[:3].mean(0), gal, v))
            add('own r1', top5(c[0], c[1:].mean(0), v))
        rows.append(r); print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
pd.set_option('display.width', 250)
df = pd.DataFrame(rows).round(2); print(df.to_string(index=False))
df.to_csv(f'{OUT}/m01_layers_{recipe}_f{k}.csv', index=False)
