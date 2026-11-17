"""A1. Variance partition with split-half reliability, per stage.

Per unit, on the 200 test images x 10 subjects x 2 halves of the 80 repetitions (subject offsets removed):
  rel    = cov over images between the two halves of the SAME subject   (reliable image-locked variance)
  shared = cov over images between DIFFERENT subjects                   (image main effect)
  inter  = rel - shared                                                 (subject x image interaction)
  noise  = var - rel ;  subj = variance of the subject offsets
Summed over units. Reported for all subjects, and for held-out vs training subjects separately
(held-out `shared` = covariance of the held-out subject with the training subjects)."""
import pandas as pd
from mi_lib import *

halves = {}
for s in range(1, 11):
    x = eeg(s, 'test', average=False)  # (200,80,63,250) fp16
    halves[s] = [T(x[:, h::2].astype(np.float32).mean(1)) for h in (0, 1)]

rows = []
for run in RUNS:
    for fold in range(1, 11):
        m = M(run, fold)
        acts = {st: [] for st in STAGES}  # stage -> [subject][half] (200, D)
        for s in range(1, 11):
            per = {st: [] for st in STAGES}
            for h in (0, 1):
                out, keep = forward_from(m, 'input', halves[s][h], keep=STAGES[1:])
                per['input'].append(halves[s][h].flatten(1))
                for st in STAGES[1:]:
                    per[st].append(keep[st].flatten(1))
            for st in STAGES:
                acts[st].append(torch.stack(per[st]))
        for st in STAGES:
            X = torch.stack(acts[st]).to(DEV)  # (10, 2, 200, D)
            mu = X.mean(2, keepdim=True)
            Xc = X - mu
            subj = mu.mean(1).var(0).sum().item()  # variance of subject offsets, summed over units
            var_s = (Xc ** 2).mean((1, 2)).sum(1)  # (10,) per-subject total variance
            rel_s = (Xc[:, 0] * Xc[:, 1]).mean(1).sum(1)  # (10,)
            Xm = Xc.mean(1)  # (10, 200, D) mean of halves
            C = torch.einsum('sid,tid->st', Xm, Xm) / X.shape[2]  # (10,10) cross-subject covariance summed over units
            u = fold - 1
            tr = [i for i in range(10) if i != u]
            off = ~torch.eye(10, dtype=torch.bool, device=DEV)
            groups = {
                'all': (var_s.mean(), rel_s.mean(), C[off].mean()),
                'heldout': (var_s[u], rel_s[u], C[u, tr].mean()),
                'train': (var_s[tr].mean(), rel_s[tr].mean(), C[tr][:, tr][~torch.eye(9, dtype=torch.bool, device=DEV)].mean()),
            }
            for g, (var, rel, shared) in groups.items():
                var, rel, shared = var.item(), rel.item(), shared.item()
                rows.append(dict(run=run, fold=fold, stage=st, group=g, var=var, subj=subj, rel=rel, shared=shared,
                                 inter=rel - shared, noise=var - rel, shared_over_rel=shared / rel, rel_over_var=rel / var))
            del X, Xc, Xm
        print(run, fold, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e01_varpart.csv', index=False)
pd.set_option('display.width', 200)
print(df.groupby(['run', 'stage', 'group'], sort=False)[['rel_over_var', 'shared_over_rel']].mean().unstack('group').round(3))
tot = df[df.group == 'all'].copy()
for c in ('subj', 'shared', 'inter', 'noise'):
    tot[c + '_frac'] = tot[c] / (tot['var'] + tot['subj'])
print(tot.groupby(['run', 'stage'], sort=False)[['subj_frac', 'shared_frac', 'inter_frac', 'noise_frac']].mean().round(3))
