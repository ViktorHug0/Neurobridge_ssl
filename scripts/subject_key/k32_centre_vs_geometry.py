"""K32: is a new subject's output "the right geometry around the wrong centre"? Measured, not assumed.
Conditions (200 test queries, normalised embeddings e against their image embeddings g):
  new own / new K     the held-out subject under its own key / under the substitution (mean training key K)
  seen own / seen wrong   two training subjects under their own key / another training subject's key
Measures:
  plain, cal      top-1 before / after the anchored re-centring (K10)
  offset_share    % of the mean squared EEG-image distance that is one constant vector: |mean(e - g)|^2 / mean|e - g|^2
  cka             linear CKA of the centred clouds: same shape, whatever the orientation
  pair_cos / other_cos   cosine of centred e_i with centred g_i / with the other images: same shape in the SAME axes
  rot, lin        top-1 after an ORACLE output map fitted with labels on 4,000 of the subject's training-image trials
                  (centred clouds; orthogonal Procrustes / ridge, strength chosen on 1,000 more), test queries centred with
                  their own mean then mapped: what is linearly recoverable beyond the centre
  key_const       % of the embedding change caused by swapping the key that is one constant vector (own vs K / own vs wrong)
Usage: k32_centre_vs_geometry.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 5000, replace=False))
FIT, VAL = slice(0, 4000), slice(4000, 5000)


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return ((a.T @ b).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())).item()


def oracle(etr, gtr, e, g, a):
    X, Y = etr - etr.mean(0), gtr - gtr.mean(0)
    ec = e - e.mean(0)
    U, _, Vt = torch.linalg.svd(X[FIT].T @ Y[FIT])
    C, scale = X[FIT].T @ X[FIT], (X[FIT].T @ X[FIT]).trace() / X.shape[1]
    maps = [torch.linalg.solve(C + lam * scale * torch.eye(len(C)), X[FIT].T @ Y[FIT]) for lam in (0.01, 0.1, 1.0, 10.0)]
    W = max(maps, key=lambda w: top1(X[VAL] @ w, nz(Y[VAL], dim=1)))
    return dict(rot=top1(ec @ (U @ Vt) + a, g), lin=top1(ec @ W + a, g))


def measures(e, g, a):
    ec, gc = e - e.mean(0), g - g.mean(0)
    S = nz(ec, dim=1) @ nz(gc, dim=1).T
    n = len(S)
    return dict(plain=top1(e, g), cal=top1(ec + a, g),
                offset_share=100 * ((e - g).mean(0).norm() ** 2 / (e - g).pow(2).sum(1).mean()).item(),
                cka=cka(e, g), pair_cos=S.diag().mean().item(), other_cos=((S.sum() - S.diag().sum()) / (n * n - n)).item())


out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, gtr, K = img_emb(m), img_emb(m, 'train'), mean_key(m)
        a, gtr = gtr.mean(0), gtr[IM]
        seen = [m.train_subs[fold % 9], m.train_subs[(fold + 4) % 9]]
        for s, t in [(fold, None)] + list(zip(seen, seen[::-1])):
            x, xtr = T(eeg(s)), T(eeg(s, 'train')[IM])
            mu, mutr = x.mean(0, keepdim=True), xtr.mean(0, keepdim=True)
            if t is None:  # new subject: own key against the substitution
                alt = lambda v, k: embed_sub(m, v, s, k, K)
            else:  # seen subject: own key against the key of another training subject
                alt = lambda v, k: m.embed(v - k + train_key(t)[None], s)
            own, swp = (nz(m.embed(x, s), dim=1), nz(m.embed(xtr, s), dim=1)), (nz(alt(x, mu), dim=1), nz(alt(xtr, mutr), dim=1))
            d = own[0] - swp[0]
            const = 100 * (d.mean(0).norm() ** 2 / d.pow(2).sum(1).mean()).item()
            for name, (e, etr) in (('own', own), ('K' if t is None else 'wrong', swp)):
                out.append(dict(run=RUN, fold=fold, cond=('new ' if t is None else 'seen ') + name, **measures(e, g, a),
                                **oracle(etr, gtr, e, g, a), key_const=const))
        print(RUN, fold, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k32_centre_vs_geometry.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby(['run', 'cond'], sort=False).mean(numeric_only=True).drop(columns='fold').round(3).to_string())
