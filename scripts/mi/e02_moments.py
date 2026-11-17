"""A2a. Moment patching: at ONE stage, match the held-out subject's activation statistics to those of a typical
training subject, then run the rest of the frozen network. Label-free.
  mean / std : per unit (flat stages) or per channel pooled over time (conv stages), stats from the 200 test trials
  cov        : whiten-colour with the full covariance, stats from the subjects' train-image activations
               (input: 63x63 channel covariance = Euclidean alignment; flat stages: unit covariance)."""
import pandas as pd
from mi_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'


def stats(a, conv):
    dims = (0, -1) if conv else (0,)
    return a.mean(dims, keepdim=True), a.std(dims, keepdim=True) + 1e-6


def msqrt(C, inv=False, shrink=0.1):
    C = (1 - shrink) * C + shrink * torch.eye(len(C), device=C.device) * C.diag().mean()
    w, V = torch.linalg.eigh(C.double())
    return ((V * w.clamp_min(1e-10).pow(-0.5 if inv else 0.5)) @ V.T).float()


def cov(a):  # a (N, D) or (N, C, T) -> covariance over D or C
    a = a.transpose(1, 2).reshape(-1, a.shape[1]) if a.dim() == 3 else a
    a = a - a.mean(0)
    return a.T @ a / len(a)


rows = []
for fold in range(1, 11):
    m = M(RUN, fold)
    g = img_emb(m)
    u = fold
    test = {s: forward_from(m, 'input', T(eeg(s)), keep=STAGES[1:])[1] | {'input': T(eeg(s))} for s in range(1, 11)}
    base = top1(test[u]['align'], g)
    rows.append(dict(fold=fold, stage='none', patch='none', acc=base))
    covs = {}
    for s in range(1, 11):  # covariance from train-image activations, flat/small stages only
        x = T(eeg(s, 'train'))
        _, k = forward_from(m, 'input', x, keep=['spatial', 'backbone', 'trunk', 'align'])
        k['input'] = x
        covs[s] = {st: cov(a.to(DEV)) for st, a in k.items()}
        del x, k
    for st in STAGES:
        conv = st in ('input', 'temporal', 'spatial')
        a = test[u][st].to(DEV)
        mu_u, sd_u = stats(a, conv)
        tr = [stats(test[s][st].to(DEV), conv) for s in m.train_subs]
        mu_t, sd_t = torch.stack([t[0] for t in tr]).mean(0), torch.stack([t[1] for t in tr]).mean(0)
        patched = {'mean': a - mu_u + mu_t, 'mean+std': (a - mu_u) / sd_u * sd_t + mu_t}
        if st in covs[u]:
            Wm = msqrt(covs[u][st], inv=True) @ msqrt(torch.stack([covs[s][st] for s in m.train_subs]).mean(0))
            c = a - mu_u
            patched['cov'] = (torch.einsum('nct,cd->ndt', c, Wm) if c.dim() == 3 else c @ Wm) + mu_t
        for name, p in patched.items():
            acc = top1(forward_from(m, st, p.cpu()) if st != 'align' else p.cpu(), g)
            rows.append(dict(fold=fold, stage=st, patch=name, acc=acc))
    print(fold, base, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e02_moments_{RUN}.csv', index=False)
print(df.pivot_table(index='stage', columns='patch', values='acc', sort=False).round(2))
