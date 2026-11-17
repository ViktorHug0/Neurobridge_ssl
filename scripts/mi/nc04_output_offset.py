"""N06 output-side person offset in the sealed regime (cached outputs of logs/holdout/f<k>, new people x never-trained
images): top-5 % at r=1/3/4, 5k and 14k galleries, after removing b x an estimate of the person's mean output:
  person  = oracle: the person's mean over ALL its trials        batch = the 64-batch mean (label-free, cohort level)
  mates   = oracle identity, deployable sample size: mean of the same person's trials inside the 64-batch"""
import pandas as pd
from nc_lib import *

k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
gals = {'5k': gallery(), '14k': gallery(big=True)}
sid = torch.as_tensor(subj)
batch = torch.arange(len(subj)) // 64
rows = []
for m in ('d6', 'rmix', 'sqfx'):
    recipe, ck = MEMBERS[m]
    y = scaled(torch.as_tensor(np.load(f'{R}/logs/holdout/f{k}/{recipe}{"" if "best" in ck else "__" + ck[11:-4]}.npy')).float().to(DEV)).cpu()
    person = torch.stack([y[sid == s].mean(0) if (sid == s).any() else y[0] * 0 for s in range(21)])[sid]
    bmean = torch.zeros(int(batch.max()) + 1, y.shape[1]).index_add_(0, batch, y) / torch.bincount(batch)[:, None]
    gid = batch * 32 + sid  # (batch, person) cell
    u, inv = torch.unique(gid, return_inverse=True)
    mates = (torch.zeros(len(u), y.shape[1]).index_add_(0, inv, y) / torch.bincount(inv)[:, None])[inv]
    for name, est in (('none', None), ('person', person), ('batch', bmean[batch]), ('mates', mates)):
        for b in ((0,) if est is None else (0.5, 1.0)):
            q = (y if est is None else y - b * est).to(DEV)
            r = dict(member=m, remove=name, b=b)
            for gn, g in gals.items():
                r.update({f'{gn} {a}': v for a, v in score(q, subj, img, g).items()})
            rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nc04_output_offset_f{k}.csv', index=False)
print(df.round(3).to_string())
