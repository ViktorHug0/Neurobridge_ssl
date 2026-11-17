"""D09 after lag alignment, do the two datasets share one route? THINGS training people (sub-21, 23, 25, 27, 29) on
never-trained images at the timing each model was trained with (baseline: raw; lag model: shifted +5), under their own
dataset key vs the Alljoined mean key vs no dataset key; and new Alljoined people under the THINGS mean key.
fold-1 rmix, epoch 8, 5k gallery, top-5 % r=1/3/4."""
import pandas as pd
from nc_lib import *
from nd03_cross_dataset_lib import shift

k = 1
gal = gallery()
K_AJ = torch.stack([key(s) for s in range(1, 21) if s not in FOLDS[k]]).mean(0).to(DEV)
K_TH = torch.stack([key(s) for s in range(21, 31)]).mean(0).to(DEV)
cells = {'TH': cell([21, 23, 25, 27, 29]), 'AJ': cell(FOLDS[k])}
rows = []
for name, recipe, lag in (('baseline', 'rmix_reg_ho5k_long_optx_things', 0), ('lag 5', 'mi_rmix_ho5k_lag5', 5)):
    f = load(recipe, k, 'checkpoint_ep08.pth')
    KT = shift(K_TH, lag)  # the THINGS key as this model sees it
    conds = {('TH', 'own key'): lambda x: shift(x, lag), ('TH', 'key -> Alljoined'): lambda x: shift(x, lag) - KT + K_AJ, ('TH', 'no dataset key'): lambda x: shift(x, lag) - KT,
             ('AJ', 'own key'): None, ('AJ', 'key -> THINGS'): lambda x: x - K_AJ + KT, ('AJ', 'no dataset key'): lambda x: x - K_AJ}
    for (c, cn), T in conds.items():
        X, subj, img = cells[c]
        y = f(X, None if T is None else (lambda a, b: T(torch.as_tensor(X[a:b]).to(DEV)) - torch.as_tensor(X[a:b]).to(DEV)))
        rows.append(dict(model=name, people={'TH': 'THINGS training people', 'AJ': 'new Alljoined people'}[c], key=cn, **score(y, subj, img, gal)))
        print(rows[-1], flush=True)
    del f; torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nd08_route.csv', index=False); print(df.round(3).to_string(index=False))
