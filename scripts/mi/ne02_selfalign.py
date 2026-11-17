"""E02 can a batch find its own cohort latency without labels? (insurance against a timing offset of the sealed cohort)
Alljoined-ONLY REVE-d6 and rmix (fold 3, ep10) on (a) five THINGS-EEG2 people = a cohort with a known +5-sample offset,
(b) the fold's held-out Alljoined people = no offset. For each candidate shift, per trial: output norm of each member and
cosine between the two members' outputs; per shift: top-5 % (5k, r=3) of the scaled sum. Then per 64-trial batch, the
shift that maximises each label-free statistic (how often it lands on the right shift)."""
from nc_lib import *
from nd03_cross_dataset_lib import shift

k = 3
fs = [load(r, k, 'checkpoint_ep10.pth') for r in ('reve_d6_ho5k_mid_optx_aj', 'rmix_reg_ho5k_mid_optx_aj')]
gal = gallery()
K_AJ = torch.stack([key(s) for s in range(1, 21) if s not in FOLDS[k]]).mean(0).to(DEV)
for name, people, lags in (('THINGS people (true offset about +5)', [21, 23, 25, 27, 29], list(range(-2, 10))), ('held-out Alljoined people (true offset 0)', FOLDS[k], list(range(-5, 6)))):
    X, subj, img = cell(people)
    X, subj, img = X[:40000], subj[:40000], img[:40000]
    stats = {s: [] for s in ('norm d6', 'norm rmix', 'agreement', 'key match')}
    print(f'== {name}', flush=True)
    for n in lags:
        T = lambda a, b: shift(torch.as_tensor(X[a:b]).to(DEV), n) - torch.as_tensor(X[a:b]).to(DEV)
        y = [f(X, T) for f in fs]
        stats['norm d6'].append(y[0].norm(dim=1)); stats['norm rmix'].append(y[1].norm(dim=1)); stats['agreement'].append((nz(y[0]) * nz(y[1])).sum(1))
        xb = shift(torch.as_tensor(X).view(-1, 64, 32, 120).mean(1).to(DEV), n)[:, :, 12:108].flatten(1); kk = K_AJ[:, 12:108].flatten()[None]
        stats['key match'].append(((xb - xb.mean(1, keepdim=True)) * (kk - kk.mean())).sum(1).cpu() / ((xb - xb.mean(1, keepdim=True)).norm(dim=1).cpu() * (kk - kk.mean()).norm().cpu()))
        print('  shift %+d: top-5 r=1/3/4 %s | norm d6 %.3f rmix %.3f | agreement %.4f | batch-mean ERP vs Alljoined key r %.3f' % (
            n, ' / '.join('%.2f' % v for v in score(scaled(y[0]) + scaled(y[1]), subj, img, gal).values()), stats['norm d6'][-1].mean(), stats['norm rmix'][-1].mean(),
            stats['agreement'][-1].mean(), stats['key match'][-1].mean()), flush=True)
    for s, v in stats.items():
        v = torch.stack(v)  # (lags, trials) or (lags, batches)
        b = v if s == 'key match' else v.view(len(lags), -1, 64).mean(2)
        pick = torch.as_tensor(lags)[b.argmax(0)]
        print('  per-batch argmax of %-10s: mean %+.2f, sd %.2f, histogram %s' % (s, pick.float().mean(), pick.float().std(), dict(zip(*[t.tolist() for t in pick.unique(return_counts=True)]))))
