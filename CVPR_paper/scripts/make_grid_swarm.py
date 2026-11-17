"""Summary of the 60 incremental settings (3 streams x 5 menu sizes x 4 numbers of repetitions) in one single-column
figure: for SATTC, SCORE with its own encoder (hollow dots, one seed), SCORE with ATM and GALA, the gain over the
zero-shot decoder of the same encoder at the end of each stream, one dot per setting (mean over 3 seeds, 10 subjects,
3 stream orders), with the mean of the 20 settings as a bar.
Reads results/things_eeg/cvpr/inc/v5/{full,skew25,skew25own}; writes CVPR_paper/figures/grid_swarm.pdf, and
grid_swarm_time.pdf: one swarm per stimulus time (the settings whose stream lasts that long, three streams pooled),
dots coloured by method.
usage: make_grid_swarm.py
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = 'results/things_eeg/cvpr/inc/v5'
STREAMS = {'uniform': 'Uniform', 'skew': 'Skewed', 'skew25': 'Partial'}
# SATTC is a slightly darker gold than in incremental.pdf (#e0a020): next to the orange of SCORE the lighter one fails
# the colour-separation check of the dataviz validator; the x labels carry the identity as well
METHODS = {'sattc': ('SATTC', '#c9a600'), 'own_score': ('SCORE', '#eb6834'), 'score': ('SCORE+ATM', '#eb6834'), 'gala': ('GALA', '#2a78d6')}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0, 'ytick.major.width': 0.6})


def swarm(y, r):
    """x offsets of a beeswarm: points are placed in order of y at the offset closest to 0 that leaves a distance
    of at least 2r (in y units, after the caller's aspect correction) to the points already placed."""
    x = np.zeros(len(y))
    for n, i in enumerate(np.argsort(y)):
        placed = np.argsort(y)[:n]
        for k in range(200):
            c = (k + 1) // 2 * (1 if k % 2 else -1) * 2 * r
            if all((c - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= (2 * r) ** 2 for j in placed):
                x[i] = c
                break
    return x


d = pd.concat([pd.read_csv(f) for p in ('full/atm*.csv', 'skew25/*.csv') for f in glob.glob(f'{R}/{p}')])
own = pd.concat([pd.read_csv(f) for p in ('full/score30s2*.csv', 'skew25own/*.csv') for f in glob.glob(f'{R}/{p}')])
d = pd.concat([d, own.assign(method='own_' + own.method)])  # SCORE with its own encoder, and that encoder's zero-shot
d = d[d.scenario.isin(STREAMS) & d.method.isin(['plain', 'own_plain', *METHODS])]
cur = d.groupby(['scenario', 'K', 'q', 'method', 't']).top1.mean()  # mean curve of each setting and method
e = cur.groupby(level=['scenario', 'K', 'q', 'method']).last().unstack('method')  # end of each stream
assert len(e) == 60, len(e)
gain = e[list(METHODS)].sub(e.plain, axis=0)
gain['own_score'] = e.own_score - e.own_plain  # against its own zero-shot decoder
best = (e.gala > e.drop(columns=['gala', 'own_plain']).max(axis=1)).groupby(level='scenario').sum()

W, H = 3.3, 2.3
fig, axs = plt.subplots(1, 3, figsize=(W, H), sharey=True)
lo, hi = gain.values.min(), gain.values.max()
r = 0.015 * (hi - lo)  # dot radius in y units
for ax, (sc, name) in zip(axs, STREAMS.items()):
    g = gain.loc[sc]
    for k, (m, (label, col)) in enumerate(METHODS.items()):
        y = g[m].values
        # 1 x unit = one method slot; convert the swarm offsets from y units to slots with the axes aspect
        aspect = (hi - lo) * 1.12 / (H * 0.74) * (W / 3 * 0.86) / len(METHODS)
        hollow = m == 'own_score'
        ax.scatter(k + swarm(y, r) / aspect, y, s=7, facecolor='white' if hollow else col, edgecolor=col if hollow else 'white',
                   linewidth=0.7 if hollow else 0.3, zorder=3)
        ax.plot([k - 0.33, k + 0.33], [y.mean()] * 2, color='#1d2340', lw=1.1, zorder=4)
    ax.axhline(0, color='#9a9a9a', lw=0.6, zorder=1)
    ax.set(xlim=(-0.6, len(METHODS) - 0.4), ylim=(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)), xticks=range(len(METHODS)))
    ax.set_xticklabels([v[0] for v in METHODS.values()], fontsize=6, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_title(f'{name} stream', fontsize=8, pad=11)
    ax.text(0.5, 1.02, f'GALA best: {best[sc]}/20', transform=ax.transAxes, ha='center', va='bottom', fontsize=6, color='#4a4f5c')
    ax.grid(axis='y', lw=0.4, color='#e4e4e4')
    ax.tick_params(axis='y', labelsize=7)
axs[0].set_ylabel('Gain over zero-shot at the\nend of the stream (points)', fontsize=7.5)
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.savefig('CVPR_paper/figures/grid_swarm.pdf')
fig.savefig('CVPR_paper/figures/grid_swarm.png', dpi=300)
print(gain.groupby(level='scenario').mean().round(1).to_string(), '\nGALA best per stream:', best.to_dict(), 'total', int(best.sum()))

# second figure: one swarm per stimulus time
def at(sc, K, q, m, minutes):
    """accuracy of a setting at this stimulus time (log-time interpolation between refits), None if the stream is shorter."""
    h = cur.loc[(sc, K, q, m)]
    mins = h.index.values * q * 0.2 / 60
    return None if minutes > mins[-1] else np.interp(np.log(minutes), np.log(mins), h.values)


W, H = 6.9, 2.5
fig = plt.figure(figsize=(W, H))
box = [0.075, 0.2, 0.915, 0.72]  # axes in figure fractions, so that the dot radius is known in both data units
ax = fig.add_axes(box)
times = [1, 5, 10, 20, None]
rows = []
for k, T in enumerate(times):
    for sc, K, q in e.index:
        if T is not None and at(sc, K, q, 'plain', T) is None:
            continue
        for m in METHODS:
            v = e.loc[(sc, K, q)] if T is None else {x: at(sc, K, q, x, T) for x in (m, 'plain', 'own_plain')}
            rows.append((k, m, v[m] - v['own_plain' if m == 'own_score' else 'plain']))
g = pd.DataFrame(rows, columns=['k', 'm', 'gain'])
lo, hi = g.gain.min(), g.gain.max()
ylim, xlim = (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)), (-0.55, len(times) - 0.45)
ypi, xpi = (ylim[1] - ylim[0]) / (H * box[3]), (xlim[1] - xlim[0]) / (W * box[2])  # data units per inch
r = 0.021 * ypi  # dot radius: 0.021 in
for k, T in enumerate(times):
    x = g[g.k == k]
    off = np.clip(swarm(x.gain.values, r) / ypi * xpi, -0.46, 0.46)
    hollow = (x.m == 'own_score').values
    col = np.array([METHODS[m][1] for m in x.m])
    ax.scatter(k + off, x.gain, s=7, facecolor=np.where(hollow, 'white', col), edgecolor=np.where(hollow, col, 'white'),
               linewidth=np.where(hollow, 0.7, 0.3), zorder=3)
ax.axhline(0, color='#9a9a9a', lw=0.6, zorder=1)
ax.set(xlim=xlim, ylim=ylim, xticks=range(len(times)))
ax.set_xticklabels([f"{'End of the stream' if T is None else f'{T} min'}\n{(g.k == k).sum() // len(METHODS)} settings" for k, T in enumerate(times)], fontsize=7.5)
ax.set_ylabel('Gain over zero-shot (points)', fontsize=8)
ax.grid(axis='y', lw=0.4, color='#e4e4e4')
ax.legend(handles=[plt.Line2D([], [], ls='', marker='o', ms=4, mfc='white' if m == 'own_score' else c, mec=c if m == 'own_score' else 'white', label=l)
                   for m, (l, c) in reversed(METHODS.items())], loc='upper left', ncol=4, fontsize=7.5, frameon=False, handletextpad=0.1, columnspacing=1.2,
          bbox_to_anchor=(0, 1.13))
fig.savefig('CVPR_paper/figures/grid_swarm_time.pdf')
fig.savefig('CVPR_paper/figures/grid_swarm_time.png', dpi=300)
print(g.groupby(['k', 'm']).gain.mean().unstack('m').round(1).to_string())

