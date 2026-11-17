"""E03 does re-rendering a NEW person through the training people's channel covariance help the frozen members?
The optx members were trained with operator transfer (x -> R_t^1/2 R_s^-1/2 x between training people), so a trial
re-rendered into a training covariance is in-distribution. Fold k, new people x never-trained images, 5k, exact ranks.
Source covariance R_s of the new trials estimated from: the person (oracle identity, all its trials), the 64-trial
batch (mixed people; stateless, deployable), or the single trial shrunk towards the population (deployable).
Target: the mean training covariance, one training person, or the average output over 4 training people.
    ne03_recolor.py <fold> <members>"""
import pandas as pd
from nc_lib import *

k = int(sys.argv[1]); members = sys.argv[2].split(',')
NEW = FOLDS[k]; TRAIN = [s for s in range(1, 21) if s not in NEW]
X, subj, img = cell(NEW)
Xt = torch.as_tensor(X)
gal = gallery()


def cov(x):  # (n, 32, 120) -> (32, 32), float64, shrunk like module.augment.cross_subject_operator_transfer
    x = x.to(DEV).double(); c = torch.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[-1])
    return .95 * c + .05 * torch.eye(32, device=DEV, dtype=torch.float64) * c.diagonal().mean()


def power(c, p):
    e, v = torch.linalg.eigh(c); return (v * e.clamp_min(1e-8) ** p) @ v.T


Rt = {}
for s in TRAIN:
    x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)[:3000])
    Rt[s] = cov(torch.as_tensor(x[np.abs(x).sum((1, 2)) > 0]))
Rbar = torch.stack(list(Rt.values())).mean(0)
Rnew = {s: cov(Xt[torch.as_tensor(subj == s)]) for s in NEW}
d = lambda a, b: (torch.linalg.matrix_norm(power(a, .5) - power(b, .5)) / torch.linalg.matrix_norm(power(b, .5))).item()
print('relative distance of R^1/2 to the mean training covariance: training people %.3f (mean), new people %s' % (
    np.mean([d(Rt[s], Rbar) for s in TRAIN]), ' '.join('%d: %.3f' % (s, d(Rnew[s], Rbar)) for s in NEW)), flush=True)
sid = torch.as_tensor(subj)


def person_op(target):  # (n, 32, 32): each trial's person covariance -> target
    ops = {s: (power(target, .5) @ power(Rnew[s], -.5)).float() for s in NEW}
    return lambda a, b: torch.stack([ops[int(s)] for s in sid[a:b]])


def batch_op(target):  # covariance of each 64-trial batch (mixed people)
    def f(a, b):
        out = []
        for c in range(a, min(b, len(Xt)), 64):
            x = Xt[c:c + 64]; out.append((power(target, .5) @ power(cov(x), -.5)).float().expand(len(x), 32, 32))
        return torch.cat(out)
    return f


def trial_op(target, g):  # each trial's own covariance shrunk towards the population by g
    def f(a, b):
        x = Xt[a:b].to(DEV).double(); c = (1 - g) * torch.einsum('nct,ndt->ncd', x, x) / 120 + g * Rbar
        e, v = torch.linalg.eigh(c)
        return (power(target, .5) @ ((v * e.clamp_min(1e-8)[:, None, :] ** -.5) @ v.transpose(1, 2))).float()
    return f


three = TRAIN[::5][:3]; four = TRAIN[1::4][:4]
conds = {'raw': None, 'person -> mean training covariance': [person_op(Rbar)], 'batch -> mean': [batch_op(Rbar)],
         'trial (shrink .5) -> mean': [trial_op(Rbar, .5)], 'trial (shrink .9) -> mean': [trial_op(Rbar, .9)],
         **{f'person -> training person {t}': [person_op(Rt[t])] for t in three},
         'person -> 4 training people, outputs averaged': [person_op(Rt[t]) for t in four],
         'batch -> 4 training people, outputs averaged': [batch_op(Rt[t]) for t in four],
         'raw + batch -> 4 training people, averaged': [None] + [batch_op(Rt[t]) for t in four]}
rows, outs = [], {}
for m in members:
    f = load(m, k)
    for cn, ops in conds.items():
        ys = []
        for op in ops or [None]:
            ys.append(scaled(f(X, None if op is None else (lambda a, b, op=op: torch.einsum('ncd,ndt->nct', op(a, b), Xt[a:b].to(DEV)) - Xt[a:b].to(DEV)))))
        outs[(m, cn)] = sum(ys) / len(ys)
        sc, h = score(outs[(m, cn)], subj, img, gal, hits=True)
        b = score(outs[(m, 'raw')], subj, img, gal, hits=True)[1]
        rows.append(dict(member=m, cond=cn, **sc, gain_r3='%+.1f +- %.1f %%' % paired(h['r3'], b['r3'])))
        print(rows[-1], flush=True)
    del f; torch.cuda.empty_cache()
if len(members) > 1:
    b = score(sum(outs[(m, 'raw')] for m in members), subj, img, gal, hits=True)[1]
    for cn in conds:
        sc, h = score(sum(outs[(m, cn)] for m in members), subj, img, gal, hits=True)
        rows.append(dict(member='+'.join(members), cond=cn, **sc, gain_r3='%+.1f +- %.1f %%' % paired(h['r3'], b['r3'])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
