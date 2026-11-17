"""Appendix figure: R2 streams for every menu size K and number of averaged repetitions q (ATM, ValCon, three seeds,
10 subjects, 3 stream orders). x = total stimulus time needed to collect the buffer: t queries x q repetitions x 200 ms
(100 ms image + 100 ms blank, THINGS-EEG-2 RSVP). Writes CVPR_paper/figures/stream_grid.pdf.
usage: make_stream_appendix.py [stream dir name, default stream50_free]
"""
import glob, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

SUB = sys.argv[1] if len(sys.argv) > 1 else 'stream50_free'
F = 'results/things_eeg/cvpr/final/atm128_vc_s33*'  # s3300 = symlink to atm128_valcon_s3300
SOA = 0.2  # seconds per repetition
STYLE = {'plain': ('Zero-shot', '#8a8a8a', '^'), 'sage_a1': ('GALA, undamped', '#eb6834', 's'),
         'full_dim': ('GALA, damped', '#2a78d6', 'o')}
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'legend.frameon': False})

s = pd.concat([pd.read_csv(f) for f in glob.glob(f'{F}/{SUB}/part_*/stream_rows.csv')])
s = s[s.method.isin(list(STYLE))]
g = s.groupby(['menu', 'reps', 'method', 't']).top1.mean()
menus, reps = sorted(s.menu.unique()), sorted(s.reps.unique())
fig, axs = plt.subplots(len(reps), len(menus), figsize=(6.9, 5.6), sharey=True)
for i, q in enumerate(reps):
    for j, k in enumerate(menus):
        ax = axs[i, j]
        for m, (name, c, mk) in STYLE.items():
            h = g.loc[(k, q, m)]
            ax.plot(h.index * q * SOA / 60, h.values, color=c, lw=1.2, marker=mk, ms=2.2, markevery=max(1, len(h) // 6),
                    label=name)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', lw=0.3, color='#e4e4e4')
        if i == 0:
            ax.set_title(f'$K$ = {k}', fontsize=8)
        if j == 0:
            ax.set_ylabel(f'$q$ = {q}\nTop-1 (%)', fontsize=7.5)
        if i == len(reps) - 1:
            ax.set_xlabel('Stimulus time (min)', fontsize=7)
axs[0, 0].legend(loc='lower right', fontsize=6, handlelength=1.4, borderaxespad=0.1)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
fig.savefig('CVPR_paper/figures/stream_grid.pdf')
fig.savefig('CVPR_paper/figures/stream_grid.png', dpi=200)
print(s.groupby('menu').t.max().to_dict(), 'refits/stream', s.groupby(['subject', 'menu', 'reps', 'seed', 'method']).t.nunique().max())
