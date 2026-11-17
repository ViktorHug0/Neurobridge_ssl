"""M02 which components of a token member work for new people, and which only for trained people?
(a) lens: the member's own readout (token mean -> head) applied after every block.
(b) knock-out: each block's attention output, feed-forward output, or one attention head is replaced by its mean over the
    batch (1,000 trials mixing people): the component keeps its average contribution and loses everything trial-specific.
Fold k, never-trained images, 5k gallery, r=3 top-5 (exact ranks), new people vs five training people; paired against the
intact member. A component whose knock-out costs seen people and not new people is person-specific machinery.
    nm02_components.py <recipe> <fold> <checkpoint> [heads | combo]"""
import pandas as pd
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = load(recipe, k, ck); m = f.model; gal = gallery()
cells = {'new': cell(FOLDS[k]), 'seen': cell(FOLDS[k % 4 + 1])}
mean_out = lambda mod, i, o: o.mean(0, keepdim=True).expand_as(o)
rows = []


def run(name, hooks=(), lens=None):
    hs = [mod.register_forward_pre_hook(fn) if pre else mod.register_forward_hook(fn) for mod, fn, pre in hooks]
    if lens is not None:  # stop after block `lens`: the later blocks become the identity
        hs += [b.register_forward_hook(lambda mod, i, o: i[0]) for b in m.blocks[lens + 1:]]
    r = dict(cond=name)
    for pn, (X, subj, img) in cells.items():
        sc, h = score(f(X, bs=1000), subj, img, gal, hits=True)
        base.setdefault(pn, h['r3'])
        r[pn] = sc['r3']; r[pn + ' r1'] = sc['r1']; r[pn + ' vs intact'] = '%+.1f +- %.1f' % paired(h['r3'], base[pn])
    for h in hs:
        h.remove()
    rows.append(r); print(r, flush=True)


if hasattr(m, 'fm'):  # braindecode REVE: layers = [attention, feed-forward], each a residual branch
    C = [{'attention': l[0], 'feed-forward': l[1]} for l in m.fm.transformer.layers]
elif hasattr(m.blocks[0], 'ff2'):  # TokenREVE
    C = [{'attention': b.out, 'feed-forward': b.ff2} for b in m.blocks]
else:  # Squeezeformer
    C = [{'attention': b.attn, 'conv': b.conv, 'feed-forward': b.ffn} for b in m.blocks]
N = len(C); kinds = list(C[0])
base = {}
run('intact')
if 'heads' in sys.argv:
    d = m.blocks[0].out.in_features // m.blocks[0].heads
    for b, blk in enumerate(m.blocks):
        for hd in range(blk.heads):
            def pre(mod, i, hd=hd):
                a = i[0].clone(); a[..., hd * d:(hd + 1) * d] = a[..., hd * d:(hd + 1) * d].mean(0, keepdim=True); return (a,)
            run(f'block {b} head {hd}', [(blk.out, pre, True)])
elif 'combo' in sys.argv:  # one kind of component in several blocks at once
    for kind in kinds:
        for bs in [tuple(range(n)) for n in range(2, N + 1)] + [tuple(range(1, N - 1))]:
            run(f'{kind} of blocks ' + ' '.join(map(str, bs)), [(C[b][kind], mean_out, False) for b in bs])
else:
    if hasattr(m.blocks[0] if hasattr(m, 'blocks') else None, 'ff2'):
        for b in range(N - 1):
            run(f'lens after block {b}', lens=b)
    for b in range(N):
        for kind in kinds:
            run(f'block {b} {kind}', [(C[b][kind], mean_out, False)])
pd.set_option('display.width', 250)
df = pd.DataFrame(rows).round(3); print(df.to_string(index=False))
df.to_csv(f'{OUT}/m02_components_{recipe}_f{k}{"_" + sys.argv[4] if len(sys.argv) > 4 else ""}.csv', index=False)
