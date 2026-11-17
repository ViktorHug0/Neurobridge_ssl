"""Why is the model destroyed by an 8 ms shift, and which part of the input carries the decision?
x_s(i) = m_s (subject's image-independent mean ERP over the 200 test images) + c_s(i) (image-specific part).
Counterfactual inputs (frozen model), seen vs held-out:
  nomean        c only                          swapmean   c + mean ERP of another training subject
  shift_mean    shift(m, 8 ms) + c              shift_c    m + shift(c, 8 ms)
  lp<f / hp>f   FFT brick-wall low-/high-pass of the whole input"""
import pandas as pd
from mi_lib import *
from e05_shift_gain import shift  # noqa: F401  (re-runs nothing: guarded below)

RUN = 'mix'
rows = []
freqs = np.fft.rfftfreq(250, 1 / 250)


def band(x, lo, hi):
    X = torch.fft.rfft(x, dim=-1)
    keep = torch.tensor((freqs >= lo) & (freqs < hi))
    return torch.fft.irfft(X * keep, n=250, dim=-1)


for fold in range(1, 11):
    m = M(RUN, fold)
    g = img_emb(m)
    xs = {s: T(eeg(s)) for s in range(1, 11)}
    for s in range(1, 11):
        x = xs[s]
        mu = x.mean(0, keepdim=True)
        c = x - mu
        other = m.train_subs[(m.train_subs.index(s) + 1) % 9] if s != fold else m.train_subs[0]
        var = {'plain': x, 'nomean': c, 'swapmean': c + xs[other].mean(0, keepdim=True),
               'shift_mean': shift(mu, 2) + c, 'shift_c': mu + shift(c, 2), 'shift_all': shift(x, 2)}
        for f in (2, 4, 6, 8, 12, 16, 20, 30, 45):
            var[f'lp<{f}'] = band(x, 0, f)
        for f in (1, 2, 4, 6, 8, 12, 16):
            var[f'hp>{f}'] = band(x, f, 999)
        for k, v in var.items():
            rows.append(dict(fold=fold, sub=s, held=s == fold, cond=k, acc=top1(forward_from(m, 'input', v), g)))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e06_mean_band_{RUN}.csv', index=False)
print(df.pivot_table(index='cond', columns='held', values='acc', sort=False).round(1))
