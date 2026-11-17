"""Epoch-matched comparison of two LOSO arms from their train.log test curves (first N epochs):
top-1 at the lowest-test-loss epoch, mean top-1 over the last 20 epochs, min test loss."""
import glob, re, sys
import numpy as np, pandas as pd

def curve(d, n):
    t = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r'top1 acc ([\d.]+)%\s+Loss: ([\d.]+)', open(d + '/train.log').read())][:n]
    a, l = np.array(t).T
    return dict(n=len(a), at_min_loss=a[l.argmin()], last20=a[-20:].mean(), min_loss=l.min(), max=a.max())

a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for d in sorted(glob.glob(a + '/*-sub-*')):
    f = re.search(r'sub-(\d+)', d).group(1)
    e = glob.glob(f'{b}/*-sub-{f}')
    if not e or len(re.findall('top1 acc', open(d + '/train.log').read())) < n:
        continue
    ca, cb = curve(d, n), curve(e[0], n)
    rows.append(dict(fold=int(f), **{'A_' + k: v for k, v in ca.items()}, **{'B_' + k: v for k, v in cb.items()}))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string()); print(df.mean().round(2).to_string())
