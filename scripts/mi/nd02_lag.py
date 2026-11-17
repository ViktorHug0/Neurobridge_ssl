"""D02 latency offset between the datasets. (a) image-specific content (per-image mean minus dataset key, 4,000 paired
training images): correlation between Alljoined and THINGS content as THINGS is shifted by a lag, posterior electrodes,
vs the same scan between two halves of each dataset (should peak at 0). (b) the occipital key time courses."""
from nc_lib import *
exec(open(f'{R}/scripts/mi/nd01_dataset_align.py').read().split("print('== 1. first order')")[0].split('from nc_lib import *')[1])
CH = json.load(open(f'{DATA}/info.json'))['ch_names']
post = [i for i, c in enumerate(CH) if c[0] in 'OP']
occ = [CH.index(c) for c in ('O1', 'Oz', 'O2')]


def lagcorr(a, b, lag):  # corr of a(t) with b(t - lag) i.e. b shifted later by lag, over a fixed window
    A, B = a[:, post, 30:90], torch.roll(b, lag, -1)[:, post, 30:90]
    A, B = A - A.mean(0), B - B.mean(0)
    return ((A * B).sum() / (A.norm() * B.norm())).item()


print('lag(samples) ms | AJ vs TH | AJa vs AJb | THa vs THb | AJa vs THa')
for lag in range(-24, 25, 2):
    print('%+3d %+5.0f | %+.4f | %+.4f | %+.4f | %+.4f' % (lag, lag * 1000 / 120, lagcorr(C['AJ'], C['TH'], lag), lagcorr(C['AJa'], C['AJb'], lag),
                                                    lagcorr(C['THa'], C['THb'], lag), lagcorr(C['AJa'], C['THa'], lag)))
t = np.arange(120) / 120 - 0.2
for g in ('AJ', 'TH'):
    kk = KEY[g][occ].mean(0).cpu().numpy()
    print(g, 'occipital key (uV-ish x100) every 2 samples from -0.2 s:', ' '.join('%+.0f' % (100 * v) for v in kk[::2]))
    print(g, 'largest |key| peaks at (ms):', [int(1000 * t[i]) for i in np.argsort(-np.abs(kk))[:6]])
