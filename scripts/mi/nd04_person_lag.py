"""D04 per-person latency: each training person's image-specific content (4,000 paired images, posterior electrodes)
against the mean content of the OTHER Alljoined people, scanned over lags; the peak (parabola-refined) is the person's
latency relative to the Alljoined group (positive = the person must be shifted later to match)."""
from nc_lib import *

d = np.load(f'{OUT}/d01_means.npz')
people = sorted(int(k[3:]) for k in d.files if k.startswith('sum'))
AJ = [s for s in people if s < 21]
CH = json.load(open(f'{DATA}/info.json'))['ch_names']
post = [i for i, c in enumerate(CH) if c[0] in 'OP']
cont = {}
for s in people:
    m = torch.tensor(d[f'sum{s}'] / np.maximum(d[f'n{s}'], 1)[:, None, None]).float().to(DEV)[:, post]
    cont[s] = m - m.mean(0)
tot = sum(cont[s] for s in AJ)
LAGS = list(range(-10, 11))
print('person | best lag (samples, ms) | corr at best | corr at 0')
res = {}
for s in people:
    ref = (tot - cont[s]) / (len(AJ) - 1) if s in AJ else tot / len(AJ)
    A = ref[:, :, 30:90]; A = A - A.mean(0)
    c = []
    for lag in LAGS:
        B = torch.roll(cont[s], lag, -1)[:, :, 30:90]; B = B - B.mean(0)
        c.append(((A * B).sum() / (A.norm() * B.norm())).item())
    i = int(np.argmax(c)); i = min(max(i, 1), len(c) - 2)
    frac = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1])
    res[s] = LAGS[i] + frac
    print('sub-%02d | %+.2f  %+4.0f ms | %.4f | %.4f' % (s, res[s], res[s] * 1000 / 120, max(c), c[LAGS.index(0)]))
a, t = np.array([res[s] for s in AJ]), np.array([res[s] for s in people if s >= 21])
print('Alljoined people: mean %+.2f, SD %.2f samples (%.0f ms), range %+.2f..%+.2f' % (a.mean(), a.std(), a.std() * 1000 / 120, a.min(), a.max()))
print('THINGS people vs the Alljoined group: mean %+.2f, SD %.2f samples, range %+.2f..%+.2f' % (t.mean(), t.std(), t.min(), t.max()))
np.save(f'{OUT}/d04_person_lag.npy', np.array([[s, res[s]] for s in people]))
