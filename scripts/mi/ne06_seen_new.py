"""E06 the same person, seen vs new: each held-out person of fold a decoded by fold a's members (new) and by another
fold's members that trained on it (seen), on the 5,000 never-trained images, 5k, exact ranks, d6 + rmix + SQF-X.
Is the seen-person advantage uniform, or do the people who are hard as new stay hard when seen?"""
from nc_lib import *
REC = {1: ('reve_d6_ho5k_long_optx_things', 'rmix_reg_ho5k_long_optx_things', 'sqf_dbn_ho5k_long_optx_things'),
       2: ('reve_d6_ho5k_mid_optx_things', 'rmix_reg_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_optx_things'),
       3: ('reve_d6_ho5k_mid_optx_things', 'rmix_reg_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_optx_things')}
CK = ('checkpoint_ep10.pth', 'checkpoint_ep10.pth', 'checkpoint_test_best.pth')
gal = gallery(); res = {}
for model_fold, people_folds in ((1, (1, 2, 3)), (3, (1, 3)), (2, (2,))):
    fs = [load(r, model_fold, c) for r, c in zip(REC[model_fold], CK)]
    for pf in people_folds:
        F = f'{R}/logs/holdout/f{pf}'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']; X = np.load(f'{F}/X.npy', mmap_mode='r')
        h = score(sum(scaled(f(X)) for f in fs), subj, img, gal, hits=True)[1]
        for s in FOLDS[pf]:
            res.setdefault(s, {})['new' if pf == model_fold else f'seen'] = 100 * h['r3'][h['subj'] == s].mean()
        print(model_fold, pf, flush=True)
    del fs; torch.cuda.empty_cache()
print('person: new -> seen (top-5 %, r=3, 5k)')
for s, v in sorted(res.items(), key=lambda kv: -kv[1]['new']):
    print('  sub-%02d  %.2f -> %s' % (s, v['new'], '%.2f (x%.2f)' % (v['seen'], v['seen'] / max(v['new'], 1e-9)) if 'seen' in v else '-'))
both = [(v['new'], v['seen']) for v in res.values() if 'seen' in v]
a, b = np.array(both).T; print('mean new %.2f, seen %.2f; r(new, seen) across people %.2f; weakest 5 as new: %.2f -> %.2f; strongest 5: %.2f -> %.2f' % (
    a.mean(), b.mean(), np.corrcoef(a, b)[0, 1], np.sort(a)[:5].mean(), b[np.argsort(a)[:5]].mean(), np.sort(a)[-5:].mean(), b[np.argsort(a)[-5:]].mean()))
