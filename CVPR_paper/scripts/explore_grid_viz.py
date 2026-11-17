"""Visualisations of the 60 incremental settings. The margin heatmap is in the paper (figures/grid_heatmap.pdf, Appendix A);
the others are drafts. Run from the repo root.
Writes CVPR_paper/figures/drafts/{A_scatter,B_heatmap,C_sorted,D_raincloud,E_dumbbell}.{pdf,png}.
margin = GALA minus the best other embedding-level method at the end of the stream."""
import glob, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

R, OUT = 'results/things_eeg/cvpr/inc/v5', 'CVPR_paper/figures/drafts'
os.makedirs(OUT, exist_ok=True)
STREAMS = {'uniform': ('Uniform', '#1baf7a'), 'skew': ('Skewed', '#6250d6'), 'skew25': ('Partial', '#e87ba4')}
GALA, SCORE, SATTC, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#c9a600', '#0b0b0b', '#898781', '#e1e0d9'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})

d = pd.concat([pd.read_csv(f) for p in ('full/atm*.csv', 'skew25/*.csv') for f in glob.glob(f'{R}/{p}')])
d = d[d.scenario.isin(STREAMS)]
e = d.groupby(['scenario', 'K', 'q', 'method', 't']).top1.mean().groupby(level=['scenario', 'K', 'q', 'method']).last().unstack('method')
assert len(e) == 60
others = [c for c in e.columns if c != 'gala']
e['best'] = e[[c for c in others if c != 'plain']].max(axis=1)
e['margin'] = e.gala - e[others].max(axis=1)
print('GALA best in', int((e.margin > 0).sum()), 'of', len(e), '| mean margin', e.margin.mean().round(1))
save = lambda fig, n: (fig.savefig(f'{OUT}/{n}.pdf'), fig.savefig(f'{OUT}/{n}.png', dpi=220), plt.close(fig))

# A. paired scatter: every setting once, GALA against SCORE with ATM
fig, ax = plt.subplots(figsize=(3.3, 3.1))
lim = (10, 95)
ax.fill_between(lim, lim, [lim[1]] * 2, color=GALA, alpha=0.06, lw=0)
ax.plot(lim, lim, color=MUTED, lw=0.8)
for sc, (name, col) in STREAMS.items():
    x = e.loc[sc]
    ax.scatter(x.score, x.gala, s=6 + x.index.get_level_values('K') / 5, color=col, edgecolor='white', linewidth=0.4, label=name, zorder=3)
ax.set(xlim=lim, ylim=lim, xlabel='SCORE with ATM, final top-1 (%)', ylabel='GALA, final top-1 (%)', aspect='equal')
n = int((e.gala > e.score).sum())
ax.text(0.04, 0.96, f'GALA higher: {n} settings', transform=ax.transAxes, va='top', fontsize=7.5, color=INK)
ax.text(0.96, 0.04, f'SCORE higher: {60 - n}', transform=ax.transAxes, ha='right', fontsize=7.5, color=INK)
ax.grid(lw=0.4, color=GRID)
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.72), fontsize=7, handletextpad=0.1, title='Stream (dot size: gallery size)', title_fontsize=6.5)
fig.tight_layout(pad=0.3)
save(fig, 'A_scatter')

# B. heatmaps of the margin: gallery size x repetitions, one panel per stream
cmap = LinearSegmentedColormap.from_list('div', ['#d03b3b', '#f0efec', GALA])
def heatmap(margin):
    v = np.abs(margin).max()
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.05), sharey=True)  # full text width
    for ax, (sc, (name, _)) in zip(axs, STREAMS.items()):
        m = margin.loc[sc].unstack('q').sort_index(ascending=False)
        im = ax.imshow(m.values, cmap=cmap, vmin=-v, vmax=v, aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f'{m.values[i, j]:+.0f}', ha='center', va='center', fontsize=8.5, color='white' if abs(m.values[i, j]) > 0.55 * v else INK)
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, fontsize=8.5)
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=8.5)
        ax.set_title(f'{name} stream', fontsize=10)
        ax.set_xlabel('Repetitions per query $q$', fontsize=9)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axs[0].set_ylabel('Gallery size $K$', fontsize=9)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    return fig


fig = heatmap(e.margin)
fig.savefig('CVPR_paper/figures/grid_heatmap.pdf')  # this one is in the paper (Appendix A)
save(fig, 'B_heatmap')

# the same under the causal protocol (inc_causal.py, a refit after every query): accuracy on the queries of the last
# 30 % of each stream, pooled; competitors = every method run on the grid (CSLS, SATTC, Hungarian, SCORE with ATM)
c = pd.concat([pd.read_csv(f) for f in glob.glob('results/things_eeg/cvpr/inc/causal1/*.csv')])
c = c[c.method.isin(['plain', 'csls', 'sattc', 'hungarian', 'score', 'gala'])]
c = c[c.t > 0.7 * c.groupby(['scenario', 'K', 'q']).t.transform('max')]
c = c.assign(hits=c.top1 * c.n).groupby(['scenario', 'K', 'q', 'method'])[['hits', 'n']].sum()
ce = (c.hits / c.n).unstack('method')
print('causal settings:', len(ce), '| streams per setting (min, max):', c.n.groupby(level=[0, 1, 2]).max().agg(['min', 'max']).tolist())
ce['margin'] = ce.gala - ce.drop(columns='gala').max(axis=1)
ce['best'] = ce.drop(columns=['gala', 'margin', 'plain']).idxmax(axis=1)
print('causal: GALA best in', int((ce.margin > 0).sum()), 'of', len(ce), '| mean margin', ce.margin.mean().round(1),
      '| per stream', {sc: (int((ce.margin.loc[sc] > 0).sum()), float(ce.margin.loc[sc].mean().round(1))) for sc in STREAMS}, '| best competitor', ce.best.value_counts().to_dict())
print(ce.round(1).to_string())
fig = heatmap(ce.margin)
fig.savefig('CVPR_paper/figures/causal_grid_heatmap.pdf')
fig.savefig('CVPR_paper/figures/causal_grid_heatmap.png', dpi=220)
plt.close(fig)

# the same for the final method (user, 2026-11-16): gala_sw_freq1 (column masses estimated from the buffer, warm start,
# stream-selected settings; inc/causal2_freq, main setting in inc/causal2/*_abl3_*) and SCORE with stream-selected settings
# (inc/causal2); CSLS, SATTC and Hungarian from causal1, on the same streams. This one is in the paper (Appendix A)
KEY = ['seed', 'subject', 'K', 'q', 'scenario', 'order']
def streams(d, methods):
    x = pd.concat([pd.read_csv(f).assign(seed=f.split('atm128_vc_s')[1][:4]) for f in glob.glob(f'results/things_eeg/cvpr/inc/{d}/atm128_vc_s*.csv')])
    return x[x.method.isin(methods)]
c2 = pd.concat([streams('causal2', ['score', 'gala_sw_freq1']), streams('causal2_freq', ['gala_sw_freq1'])]).replace({'method': {'gala_sw_freq1': 'gala_sw'}})
c2 = pd.concat([c2, streams('causal1', ['csls', 'sattc', 'hungarian']).merge(c2.loc[c2.method == 'gala_sw', KEY].drop_duplicates())])
c2 = c2[c2.t > 0.7 * c2.groupby(['scenario', 'K', 'q']).t.transform('max')]
print('final: streams per setting (min, max):', c2[c2.method == 'gala_sw'].drop_duplicates(KEY).groupby(['scenario', 'K', 'q']).size().agg(['min', 'max']).tolist())
c2 = c2.assign(hits=c2.top1 * c2.n).groupby(['scenario', 'K', 'q', 'method'])[['hits', 'n']].sum()
c2 = (c2.hits / c2.n).unstack('method')
c2['margin'] = c2.gala_sw - c2.drop(columns='gala_sw').max(axis=1)
print('final: GALA best in', int((c2.margin > 0).sum()), 'of', len(c2), '| mean margin', c2.margin.mean().round(1),
      '| per stream', {sc: (int((c2.margin.loc[sc] > 0).sum()), float(c2.margin.loc[sc].mean().round(1))) for sc in STREAMS})
print(c2.round(1).to_string())
fig = heatmap(c2.margin)
fig.savefig('CVPR_paper/figures/final_grid_heatmap.pdf')
fig.savefig('CVPR_paper/figures/final_grid_heatmap.png', dpi=220)
plt.close(fig)

# C. the 60 margins sorted, as thin bars
fig, ax = plt.subplots(figsize=(3.3, 1.9))
s = e.margin.sort_values(ascending=False)
ax.bar(range(60), s.values, width=0.72, color=[STREAMS[i[0]][1] for i in s.index])
ax.axhline(0, color=INK, lw=0.6)
ax.set(xlim=(-1, 60), xticks=[], ylabel='GALA minus best\ncompetitor (points)')
ax.spines['bottom'].set_visible(False)
ax.text(25.5, s.max() * 0.82, f'GALA best in {int((s > 0).sum())} of 60 settings', ha='center', fontsize=7.5)
ax.grid(axis='y', lw=0.4, color=GRID)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n_) for n_, c in STREAMS.values()], loc='upper right', fontsize=7, handlelength=0.9, handletextpad=0.4)
fig.tight_layout(pad=0.3)
save(fig, 'C_sorted')

# D. raincloud: density and dots of the margin, one row per stream
fig, ax = plt.subplots(figsize=(3.3, 1.9))
xs = np.linspace(e.margin.min() - 3, e.margin.max() + 3, 300)
for k, (sc, (name, col)) in enumerate(STREAMS.items()):
    m, y0 = e.loc[sc].margin.values, 2 - k
    dens = gaussian_kde(m, 0.35)(xs)
    ax.fill_between(xs, y0 + 0.08, y0 + 0.08 + 0.55 * dens / dens.max(), color=col, alpha=0.35, lw=0)
    ax.scatter(m, y0 - 0.12 + 0.12 * np.sin(np.arange(len(m)) * 2.4), s=9, color=col, edgecolor='white', linewidth=0.3, zorder=3)
    ax.plot([m.mean()] * 2, [y0 - 0.3, y0 + 0.62], color=INK, lw=1.0)
    ax.text(xs[-1], y0 + 0.36, f'{int((m > 0).sum())}/20', ha='right', va='center', fontsize=7.5)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set(yticks=[2, 1, 0], ylim=(-0.45, 2.8), xlabel='GALA minus best competitor, end of stream (points)')
ax.set_yticklabels([v_[0] for v_ in STREAMS.values()])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
fig.tight_layout(pad=0.3)
save(fig, 'D_raincloud')

# E. dumbbells: every setting as a row, SATTC, SCORE with ATM and GALA as gains over zero-shot
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6), sharex=True)
for ax, (sc, (name, _)) in zip(axs, STREAMS.items()):
    x = e.loc[sc].sort_values('margin')
    g = x[['sattc', 'score', 'gala']].sub(x.plain, axis=0)
    for i, (idx, r) in enumerate(g.iterrows()):
        ax.plot([r.score, r.gala], [i, i], color=GALA if r.gala > r.score else SCORE, lw=1.6, alpha=0.45, solid_capstyle='round')
    ax.scatter(g.sattc, range(20), s=7, color=SATTC, zorder=3, label='SATTC')
    ax.scatter(g.score, range(20), s=11, color=SCORE, edgecolor='white', linewidth=0.3, zorder=4, label='SCORE with ATM')
    ax.scatter(g.gala, range(20), s=11, color=GALA, edgecolor='white', linewidth=0.3, zorder=5, label='GALA')
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(20)); ax.set_yticklabels([f'{K}, {q}' for K, q in g.index], fontsize=4.8)
    ax.tick_params(axis='y', length=0)
    ax.set_title(f'{name} stream', fontsize=8)
    ax.grid(axis='x', lw=0.4, color=GRID)
    ax.set_xlabel('Gain over zero-shot (points)', fontsize=7)
axs[0].set_ylabel('Setting ($K$, $q$), sorted by margin', fontsize=7)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc='lower center', ncol=3, fontsize=7.5, handletextpad=0.1, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.06, 1, 1))
save(fig, 'E_dumbbell')

# contact sheet for a quick look
from PIL import Image
ims = [Image.open(f'{OUT}/{n}.png') for n in ('A_scatter', 'B_heatmap', 'C_sorted', 'D_raincloud')]
w = max(i.width for i in ims[:2]) + max(i.width for i in ims[2:]) + 60
hgt = max(ims[0].height + ims[1].height, ims[2].height + ims[3].height) + 60
sheet = Image.new('RGB', (w, hgt), 'white')
sheet.paste(ims[0], (20, 20)); sheet.paste(ims[1], (20, 40 + ims[0].height))
x2 = 40 + max(i.width for i in ims[:2])
sheet.paste(ims[2], (x2, 20)); sheet.paste(ims[3], (x2, 40 + ims[2].height))
sheet.save(f'{OUT}/contact_ABCD.png')
