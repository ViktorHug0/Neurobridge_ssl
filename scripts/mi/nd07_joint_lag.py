"""D07 how does the JOINT model (rmix trained on Alljoined + THINGS-EEG2, fold 1, ep10) hold the two timings?
never-trained images, 5k gallery, top-5 % r=1/3/4.
  held-out Alljoined people, shifted earlier by n (towards THINGS timing): does a THINGS-timed route exist for them?
  THINGS training people (seen people), raw vs shifted later by n (towards Alljoined timing), +- the Alljoined mean key:
  is their decoding tied to their own timing?"""
import pandas as pd
from nc_lib import *
from nd03_cross_dataset_lib import shift

k, member = 1, sys.argv[1] if len(sys.argv) > 1 else 'rmix'
f = load(member, k)
gal = gallery()
K_AJ = torch.stack([key(s) for s in range(1, 21) if s not in FOLDS[k]]).mean(0).to(DEV)
K_TH = torch.stack([key(s) for s in range(21, 31)]).mean(0).to(DEV)
rows = []


def run(name, X, subj, img, T):
    y = f(X, None if T is None else (lambda a, b: T(torch.as_tensor(X[a:b]).to(DEV)) - torch.as_tensor(X[a:b]).to(DEV)))
    rows.append(dict(cond=name, **score(y, subj, img, gal))); print(rows[-1], flush=True)


X, subj, img = cell(FOLDS[k])
for n in (0, -2, -4, -5, -6, -8):
    run(f'new Alljoined people, shift {n:+d}', X, subj, img, None if n == 0 else (lambda x, n=n: shift(x, n)))
run('new Alljoined people, shift -5, key -> K_TH', X, subj, img, lambda x: shift(x - K_AJ, -5) + K_TH)
del X
X, subj, img = cell([21, 23, 25, 27, 29])
for n in (0, 2, 5, 8):
    run(f'THINGS training people, shift {n:+d}', X, subj, img, None if n == 0 else (lambda x, n=n: shift(x, n)))
run('THINGS training people, shift +5, key -> K_AJ', X, subj, img, lambda x: shift(x - K_TH, 5) + K_AJ)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nd07_joint_lag_{member}.csv', index=False); print(df.round(3).to_string(index=False))
