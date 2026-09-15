"""Standalone figure from audited JSON; no training dependencies."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
data = json.loads((here / 'analysis.json').read_text())
means = data['means']
ids = list(means)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
colors = ['#404040', '#2c7fb8'] + ['#7b3294'] * 6 + ['#d95f02', '#16865e']
x = np.arange(10)
ax = axes[0]
ax.plot(x, [means[c]['top1_fused'] for c in ids], 'o-', color='#176b91', label='Fused')
ax.plot(x, [means[c]['top1_atm'] for c in ids], 's--', color='#c77c26', label='ATM branch')
ax.plot(x, [means[c]['top1_ts'] for c in ids], '^:', color='#777777', label='TS branch')
ax.set(xticks=x, xticklabels=ids, ylabel='Top-1 accuracy (%)', title='Three-fold screening means')
ax.legend(frameon=False, fontsize=9)
for ax, key, label, title in [(axes[1], 'eeg_params', 'Unique EEG-side parameters (thousands)', 'Parameter storage'),
                            (axes[2], 'epoch_seconds', 'Training epoch time (seconds)', 'Training compute')]:
    for c, color in zip(ids, colors):
        xval = means[c][key] / (1000 if key == 'eeg_params' else 1)
        yval = means[c]['top1_fused']
        ax.scatter(xval, yval, color=color, s=42, zorder=3)
        offsets = {'C1':(-17,8),'C2':(4,6),'C3':(-20,-13),'C4':(3,6),
                   'C5':(3,-12),'C6':(-21,-11),'C7':(-21,7),'C8':(3,-12),
                   'C9':(3,7),'C0':(3,7)}
        if key == 'epoch_seconds':
            offsets.update(C4=(8, 13), C2=(8, -5))
        ax.annotate(c, (xval,yval), xytext=offsets[c], textcoords='offset points', fontsize=9)
    ax.set(xlabel=label, ylabel='Fused top-1 accuracy (%)', title=title)
    ax.margins(x=.16, y=.2)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle('Tiny TSConv + ATM sharing screen | one seed, 128D, pairwise mixup\n'
             'Inner validation; checkpoints selected by mean branch validation loss', fontsize=12)
fig.savefig(here / 'overview.png', dpi=200)
fig.savefig(here / 'overview.pdf')
