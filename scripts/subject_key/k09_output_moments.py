"""K09: output-side moment matching on top of the key substitution (ATM), label-free and gallery-free.
The new subject's 200 embeddings are shifted (and scaled per dimension) so that their mean (and std) equal those of
the TRAINING subjects' embeddings on training images (4-repetition averages; no test image is used for the reference).
  raw: moments of the projector output; unit: moments of the L2-normalised embeddings. Also own-mean centring alone.
Usage: k09_output_moments.py <run>"""
import pandas as pd
from sk_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
rng = np.random.default_rng(0)
rows = []
for fold in range(1, 11):
    m = load(RUN, fold)
    g = img_emb(m)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); K = mean_key(m)
    im = np.sort(rng.choice(16540, 1500, replace=False))
    ref = torch.cat([m.embed(T(eeg(s, 'train')[im]), s) for s in m.train_subs])
    r = dict(fold=fold)
    for name, e in (('own', m.embed(x, fold)), ('sub', embed_sub(m, x, fold, mu, K))):
        r[name] = top1(e, g)
        for space, f in (('raw', lambda v: v), ('unit', lambda v: nz(v, dim=1))):
            a, b = f(e), f(ref)
            r[f'{name}_{space}_mean'] = top1(a - a.mean(0) + b.mean(0), g)
            r[f'{name}_{space}_meanstd'] = top1((a - a.mean(0)) / a.std(0) * b.std(0) + b.mean(0), g)
        r[f'{name}_centre'] = top1(nz(e, dim=1) - nz(e, dim=1).mean(0), g)
    rows.append(r)
    print({k: round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k09_output_moments_{RUN.replace("/", "_")}.csv', index=False)
pd.set_option('display.width', 250)
print(df.mean().round(2).to_frame().T.to_string(index=False))
