"""N03: what is learned late? REVE-d6 (ho5k, fold 1) checkpoints ep 4..16: the four person x image cells under the own key,
and seen people under a wrong key (another training person's). top-5 % r=3, 5k gallery."""
import pandas as pd
from nc_lib import *

k, member = 1, sys.argv[1] if len(sys.argv) > 1 else 'd6'
EPOCHS = [int(e) for e in sys.argv[2].split(',')] if len(sys.argv) > 2 else [4, 6, 8, 10, 12, 14, 16]
NEW, SEENP = FOLDS[k], FOLDS[2]
TRAIN = [s for s in range(1, 21) if s not in NEW]
keys = {s: key(s).to(DEV) for s in range(1, 21)}
own = torch.stack([keys.get(s, keys[1]) for s in range(21)])
wrong = own.clone()
for i, s in enumerate(SEENP):
    wrong[s] = keys[[t for t in TRAIN if t != s][(3 * i + 1) % 14]]
cells = {(pn, im): cell(p, I) for pn, p in (('new', NEW), ('seen', SEENP)) for im, I in (('held', HELD), ('trained', SEEN))}
gal = {'held': gallery(HELD), 'trained': gallery(SEEN)}
rows = []
for ep in EPOCHS:
    f = load(member, k, f'checkpoint_ep{ep:02d}.pth')
    r = dict(epoch=ep)
    for (pn, im), (X, subj, img) in cells.items():
        r[f'{pn}/{im}'] = score(f(X), subj, img, gal[im])['r3']
        if pn == 'seen':
            sid = torch.as_tensor(subj).to(DEV)
            r[f'{pn}/{im}/wrong'] = score(f(X, lambda a, b: (wrong - own)[sid[a:b]]), subj, img, gal[im])['r3']
    rows.append(r); print(r, flush=True)
    del f; torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nc02_epochs_{member}.csv', index=False)
print(df.round(3).to_string())
