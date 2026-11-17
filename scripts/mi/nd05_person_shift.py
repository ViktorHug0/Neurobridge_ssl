"""D05 are held-out Alljoined people latency outliers? Per held-out person, top-5 % (r=3, 5k gallery, never-trained
images) as its trials are shifted by n samples (n > 0 = later), next to a label-free lag estimate: the lag at which the
person's key (mean ERP) best matches the mean training key (posterior electrodes, 0-0.6 s).
    nd05_person_shift.py <fold> <member>"""
import pandas as pd
from nc_lib import *
from nd03_cross_dataset_lib import shift

k, member = int(sys.argv[1]), sys.argv[2]
f = load(member, k)
gal = gallery()
X, subj, img = cell(FOLDS[k])
CH = json.load(open(f'{DATA}/info.json'))['ch_names']
post = [i for i, c in enumerate(CH) if c[0] in 'OP']
K = torch.stack([key(s) for s in range(1, 21) if s not in FOLDS[k]]).mean(0)[post]
LAGS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
rows = {}
for n in LAGS:
    _, h = score(f(X, None if n == 0 else (lambda a, b: shift(torch.as_tensor(X[a:b]).to(DEV), n) - torch.as_tensor(X[a:b]).to(DEV))), subj, img, gal, hits=True)
    rows[n] = {s: 100 * h['r3'][h['subj'] == s].mean() for s in FOLDS[k]} | {'all': 100 * h['r3'].mean()}
    print(n, {a: round(b, 2) for a, b in rows[n].items()}, flush=True)
df = pd.DataFrame(rows).T; df.index.name = 'shift'
est = {}
for s in FOLDS[k]:
    m = key(s)[post]
    c = [torch.corrcoef(torch.stack([K[:, 24:96].flatten(), shift(m, n)[:, 24:96].flatten()]))[0, 1].item() for n in range(-6, 7)]
    est[s] = int(np.argmax(c)) - 6
print(df.round(2).to_string())
print('best shift per person      :', {s: int(df[s].idxmax()) for s in FOLDS[k]})
print('label-free key-lag estimate:', est)
print('pooled: no shift %.3f | oracle per-person best %.3f | key-lag estimate %.3f' % (
    df['all'][0], np.mean([df[s].max() for s in FOLDS[k]]), np.mean([df[s][int(np.clip(est[s], -4, 4))] for s in FOLDS[k]])))
df.to_csv(f'{OUT}/nd05_person_shift_f{k}_{member}.csv')
