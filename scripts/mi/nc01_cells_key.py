"""N01 + N02, fold-1 frozen members, 5k gallery, top-5 % at r=1/3/4.
N01  person x image 2x2: new/seen person x never-trained (HELD) / trained (SEEN) images.
N02  key dependence on single trials (HELD images). key = the person's mean ERP over all its training-set trials (oracle,
     ~60k trials). K = mean key of the 15 Alljoined training people.
       seen person: own | wrong (another training person's key) | none (x - m_s) | K (x - m_s + K)
       new person : own | none | K | mid (x + .5 (K - m_u)) | nearest training key"""
import pandas as pd
from nc_lib import *

k = int(os.environ.get('K', 1))
NEW = FOLDS[k]
SEENP = FOLDS[k % 4 + 1]  # 5 training people of the fold-k models (the next fold's), dealt to match difficulty
TRAIN = [s for s in range(1, 21) if s not in NEW]
keys = {s: key(s).to(DEV) for s in range(1, 21)}
K = torch.stack([keys[s] for s in TRAIN]).mean(0)
members = sys.argv[1].split(',') if len(sys.argv) > 1 else ['d6', 'rmix', 'sqfx']
cells = {(pn, im): cell(p, I) for pn, p in (('new', NEW), ('seen', SEENP)) for im, I in (('held', HELD), ('trained', SEEN))}
gal = {'held': gallery(HELD), 'trained': gallery(SEEN)}
rows = []
for mname in members:
    f = load(mname, k)
    for (pn, im), (X, subj, img) in cells.items():
        rows.append(dict(member=mname, exp='N01', person=pn, images=im, key='own', **score(f(X), subj, img, gal[im])))
        print(rows[-1], flush=True)
    for pn in ('seen', 'new'):
        X, subj, img = cells[(pn, 'held')]
        sid = torch.as_tensor(subj).to(DEV)
        own = torch.stack([keys[s] if s in keys else K for s in range(21)])  # (21, 32, 120) by person id
        conds = {'none': -own, 'K': K - own}
        if pn == 'seen':
            wrong = own.clone()
            for i, s in enumerate(SEENP):
                wrong[s] = keys[[t for t in TRAIN if t != s][(3 * i + 1) % 14]]
            conds['wrong'] = wrong - own
        else:
            conds['mid'] = 0.5 * (K - own)
            near = own.clone()
            for s in NEW:
                near[s] = min((keys[t] for t in TRAIN), key=lambda m: (m - keys[s]).norm().item())
            conds['nearest'] = near - own
        for cn, D in conds.items():
            y = f(X, lambda a, b: D[sid[a:b]])
            rows.append(dict(member=mname, exp='N02', person=pn, images='held', key=cn, **score(y, subj, img, gal['held'])))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/nc01_f{k}_{"_".join(members)}.csv', index=False)
    del f; torch.cuda.empty_cache()
print(pd.DataFrame(rows).round(3).to_string())
kn = torch.stack([keys[s] for s in range(1, 21)])
print('key norm (mean over people)', kn.flatten(1).norm(dim=1).mean().item(), '| between-person key sd norm', (kn - kn.mean(0)).flatten(1).norm(dim=1).mean().item(),
      '| single-trial norm', float(np.linalg.norm(cells[("new", "held")][0][:2000].reshape(2000, -1), axis=1).mean()))
