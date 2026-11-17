"""Paper figure 3 from the final-model CSVs. Writes CVPR_paper/figures/regimes.pdf: (a) R2 stream, buffer top-1 per
refit; (b) R3 open gallery; both ATM, ValCon checkpoints, three seeds.
usage: make_figures.py
"""
import glob, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

F = 'results/things_eeg/cvpr/final/atm128_vc_s33*'  # s3300 = symlink to atm128_valcon_s3300
OUT = 'CVPR_paper/figures'
# categorical slots 1-3 (dataviz reference palette, validated light); markers carry identity in print / CVD
STYLE = {'sage': ('GALA', '#2a78d6', 'o'), 'score': ('SCORE', '#eb6834', 's'), 'plain': ('Zero-shot', '#1baf7a', '^'),
         'sage_a1': ('GALA, undamped', '#eb6834', 's'), 'full_dim': ('GALA, damped', '#2a78d6', 'o')}
STYLE['plain'] = ('Zero-shot', '#8a8a8a', '^')  # v2: SCORE no longer drawn; grey zero-shot reference
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})


def line(ax, x, y, key, label_last=True):
    name, c, m = STYLE[key]
    ax.plot(x, y, color=c, marker=m, ms=4, lw=1.5, label=name)


os.makedirs(OUT, exist_ok=True)
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7))  # two-column, as Fig. 2
# (a) R2 stream, three seeds. x = refit rank (25 per stream): every (menu K, repetitions q, order) weighs equally at
# each point. Binning by t/K instead mixes menu sizes and makes even the plain decoder drift.
s = pd.concat([pd.read_csv(f) for f in glob.glob(f'{F}/stream50_free/part_*/stream_rows.csv')])
s = s[s.method.isin(['plain', 'sage_a1', 'full_dim'])]
s['refit'] = s.groupby(['subject', 'menu', 'reps', 'seed', 'method']).t.rank(method='dense')
g = s.groupby(['method', 'refit']).top1.mean()
for key in ('plain', 'sage_a1', 'full_dim'):
    h = g.loc[key]
    axs[0].plot(h.index, h.values, color=STYLE[key][1], marker=STYLE[key][2], ms=3.5, lw=1.6, markevery=8,
                label=STYLE[key][0])
axs[0].set(xlabel='Refit', ylabel='Top-1 (%)', xticks=[1, 10, 20, 30, 40, 50], ylim=(0, 100))
axs[0].set_title('(a) Streaming (R2)', fontsize=9.5)
axs[0].legend(loc='lower right', fontsize=8, handlelength=1.8, borderaxespad=0.2)
# (b) R3 open gallery, three seeds
R = F
d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/regimes/part_*/regime_rows.csv')])
d = d[d.regime == 'R4'].groupby(['level', 'method']).top1.mean()
for key in ('plain', 'sage'):
    r = d.xs(key, level='method')
    axs[1].plot(r.index, r.values, color=STYLE[key][1], marker=STYLE[key][2], ms=4, lw=1.6, label=STYLE[key][0])
# GALA + support dropped from the figure (user, 2026-10-25)
axs[1].set(xlabel='Distractors $D$', ylabel='Top-1 (%)', xticks=[0, 200, 400, 800, 1654], ylim=(0, 100))
axs[1].set_title('(b) Open gallery (R3)', fontsize=9.5)
axs[1].legend(loc='upper right', fontsize=8, handlelength=1.8, borderaxespad=0.2)
for ax in axs:
    ax.grid(axis='y', lw=0.4, color='#dddddd')
fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig(f'{OUT}/regimes.pdf')
fig.savefig(f'{OUT}/regimes.png', dpi=200)
print(g.unstack(0).iloc[[0, -1]].round(1), d.unstack().round(1), sep='\n')
