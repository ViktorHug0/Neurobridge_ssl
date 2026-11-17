"""Input-level counterfactuals on the held-out subject (frozen model): global time shift and amplitude gain.
TSConv reads 35 time bins with bin-specific weights, so a latency offset between the new subject and the
training population would misroute the code. accuracy(shift, gain) per fold; also for seen subjects as control.
shift > 0 delays the signal (edge-padded). 1 sample = 4 ms."""
import pandas as pd
from mi_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'
SHIFTS = list(range(-12, 13, 2))
GAINS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]


def shift(x, k):
    if k == 0:
        return x
    pad = x[..., :1].expand(*x.shape[:-1], abs(k)) if k > 0 else x[..., -1:].expand(*x.shape[:-1], abs(k))
    return torch.cat([pad, x[..., :-k]], -1) if k > 0 else torch.cat([x[..., -k:], pad], -1)


if __name__ == '__main__':
    rows = []
    for fold in range(1, 11):
        m = M(RUN, fold)
        g = img_emb(m)
        for s in range(1, 11):
            x = T(eeg(s))
            for k in SHIFTS:
                rows.append(dict(fold=fold, sub=s, held=s == fold, shift=k, gain=1.0, acc=top1(forward_from(m, 'input', shift(x, k)), g)))
            for a in GAINS:
                rows.append(dict(fold=fold, sub=s, held=s == fold, shift=0, gain=a, acc=top1(forward_from(m, 'input', a * x), g)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e05_shift_gain_{RUN}.csv', index=False)
    pd.set_option('display.width', 250)
    sh = df[df.gain == 1.0]
    print(sh.pivot_table(index='held', columns='shift', values='acc').round(1))
    print(df[df['shift'] == 0].pivot_table(index='held', columns='gain', values='acc').round(1))
    h = sh[sh.held].pivot_table(index='fold', columns='shift', values='acc')
    print(h.round(1))
    print('held-out: plain', h[0].mean(), ' oracle best shift', h.max(1).mean(), ' best shifts', h.idxmax(1).tolist())
