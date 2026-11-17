"""Output-space diagnostics, held-out vs seen subjects (same checkpoint, 200 test images).
How much of the held-out deficit is a gallery prior (hubness / constant offset) vs pair-specific misalignment?
  top1          plain cosine
  center        subtract the subject's own mean embedding (label-free)
  colcenter     subtract each gallery column's mean score (label-free)
  oracle_bias   best additive per-gallery-image bias, fitted on the test labels themselves (upper bound of ANY prior fix)
  hub10         share of queries whose top-1 prediction is one of the subject's 10 most-predicted images (uniform = 5%)
  pos / neg     mean cosine to the correct image / to the other 199
  rank1_gap     pos minus the best wrong score, mean
"""
import pandas as pd
from mi_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'
nz = torch.nn.functional.normalize
rows = []
for fold in range(1, 11):
    m = M(RUN, fold)
    g = img_emb(m)
    for s in range(1, 11):
        e = forward_from(m, 'input', T(eeg(s)))
        S = nz(e, dim=1) @ g.T
        y = torch.arange(200)
        acc = lambda S: 100 * (S.argmax(1) == y).float().mean().item()
        c = torch.zeros(200, requires_grad=True)
        opt = torch.optim.Adam([c], lr=0.05)
        for _ in range(300):
            loss = torch.nn.functional.cross_entropy((S + c) * 50, y)
            opt.zero_grad(); loss.backward(); opt.step()
        pred = S.argmax(1)
        counts = torch.bincount(pred, minlength=200).sort(descending=True).values
        pos = S.diag()
        off = S.masked_fill(torch.eye(200, dtype=torch.bool), -9)
        rows.append(dict(fold=fold, sub=s, held=s == fold, top1=acc(S), center=top1(e - e.mean(0), g),
                         colcenter=acc(S - S.mean(0)), oracle_bias=acc(S + c.detach()),
                         hub10=100 * counts[:10].sum().item() / 200, pos=pos.mean().item(),
                         neg=(S.sum() - pos.sum()).item() / (200 * 199), rank1_gap=(pos - off.max(1).values).mean().item(),
                         norm=e.norm(dim=1).mean().item(), mean_norm=e.mean(0).norm().item()))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e04_geometry_{RUN}.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby('held').mean(numeric_only=True).drop(columns=['fold', 'sub']).round(3).T)
