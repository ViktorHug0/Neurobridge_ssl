"""The subject key. x_s(i) = m_s + c_s(i): m_s = image-independent mean ERP of subject s ("key"), c_s = image-specific part.
A. key-swap matrix: accuracy of data c_s under key m_t, all 10x10 pairs (+ key = 0, key = mean of the training keys).
B. linearisation: is the model a key-indexed linear decoder?  e ~ f(m_t) + J(m_t) c_s   (JVP at the key)
C. localisation: run data s under a WRONG key, then restore the right key's mean activation at one sub-stage only and
   continue. Every nonlinearity before that sub-stage was gated by the wrong key; the drop between two successive
   sub-stages is the part of the key effect implemented by the nonlinearity between them."""
import pandas as pd
from torch.func import jvp
from mi_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'


def chain(m):
    e = m.enc
    return [('t_lin', lambda x: e.tsconv[:3](x.unsqueeze(1))), ('t_act', e.tsconv[3]), ('s_lin', e.tsconv[4:6]),
            ('s_act', e.tsconv[6]), ('bb', lambda x: e.projection(x).flatten(1)), ('fc1', e.proj_eeg[0]),
            ('res', e.proj_eeg[1]), ('ln', e.proj_eeg[2:]), ('align', m.eeg_proj)]


def run_grad(m, x, start=0, keep=False):
    acts = []
    for name, f in chain(m)[start:]:
        x = f(x)
        if keep:
            acts.append(x)
    return acts if keep else x


run = torch.no_grad()(run_grad)


A, B, C = [], [], []
for fold in range(1, 11):
    m = M(RUN, fold)
    g = img_emb(m)
    xs = {s: T(eeg(s)).to(DEV) for s in range(1, 11)}
    mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
    c = {s: xs[s] - mu[s] for s in xs}
    keys = dict(mu) | {'zero': torch.zeros_like(mu[1]), 'trainmean': torch.stack([mu[t] for t in m.train_subs]).mean(0)}
    names = [n for n, _ in chain(m)]
    f_full = lambda x: run_grad(m, x)
    for s in range(1, 11):
        for t, k in keys.items():
            A.append(dict(fold=fold, data=s, key=t, held=s == fold, acc=top1(run(m, c[s] + k).cpu(), g)))
        for t in ([s] + [m.train_subs[0] if s != m.train_subs[0] else m.train_subs[1]]):
            out, tang = jvp(f_full, (mu[t].expand(200, -1, -1).contiguous(),), (c[s],))
            B.append(dict(fold=fold, data=s, key=t, held=s == fold, same=s == t, lin=top1((out + tang).detach().cpu(), g),
                          tangent_only=top1(tang.detach().cpu(), g), true=top1(run(m, c[s] + mu[t]).cpu(), g)))
        if s == fold:
            continue
        right = run(m, xs[s], keep=True)
        for t in [x for x in m.train_subs if x != s][:3]:
            wrong = run(m, c[s] + mu[t], keep=True)
            C.append(dict(fold=fold, data=s, key=t, stage='none', acc=top1(wrong[-1].cpu(), g)))
            for j, n in enumerate(names):
                a = wrong[j] - wrong[j].mean(0, keepdim=True) + right[j].mean(0, keepdim=True)
                C.append(dict(fold=fold, data=s, key=t, stage=n, acc=top1((run(m, a, start=j + 1) if j + 1 < len(names) else a).cpu(), g)))
    print(fold, flush=True)
A, B, C = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)
for n, d in zip('ABC', (A, B, C)):
    d.to_csv(f'{OUT}/e07_key_{n}_{RUN}.csv', index=False)
pd.set_option('display.width', 250)
A['kind'] = np.where(A.key.astype(str) == A.data.astype(str), 'own', np.where(A.key.astype(str) == A.fold.astype(str), 'heldout_key',
                     np.where(A.key.isin(['zero', 'trainmean']), A.key, 'other_train_key')))
print('A. accuracy by (data is held-out?, key kind)\n', A.pivot_table(index='held', columns='kind', values='acc').round(1))
h = A[A.held & ~A.key.isin(['zero', 'trainmean'])].pivot_table(index='fold', columns='key', values='acc')
print('A. held-out data under each key (rows = fold)\n', h.round(1))
print('   own key', np.mean([h.loc[f, f] for f in h.index]).round(2), ' best training key', np.mean([h.loc[f].drop(f).max() for f in h.index]).round(2))
print('B. linearisation at the key\n', B.groupby(['held', 'same'])[['true', 'lin', 'tangent_only']].mean().round(1))
print('C. restore the right mean activation at one sub-stage (seen subjects under a wrong key)\n',
      C.groupby('stage', sort=False).acc.mean().round(1))
