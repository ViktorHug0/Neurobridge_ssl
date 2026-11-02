"""E4.3 streaming figures (R2): accuracy vs recording time, one row per menu size K.

Columns: buffer (every trial seen so far, under the latest fit) | cumulative prequential, transductive at arrival
(`@new`: each chunk decoded by the fit that first included it) | cumulative prequential, strict-causal (`@prev`: each
chunk decoded by the map frozen before it arrived). Cumulative = running mean over the chunks seen so far (equal-size
chunks), i.e. the accuracy the user has experienced up to time x; it ends at the table's prequential number.
x = recording time = trials seen x 0.2 s (THINGS-EEG-2 RSVP, 5 Hz); a q-repetition query costs 0.2 q s, so every q
ends at K x 80 x 0.2 s. Hue = method (zero-shot grey, thin), tint = q (lighter = fewer repetitions), line style =
encoder (solid ATM-128, dotted SAMGA = SCORE's own stack). Means over subjects, stream orders and seeds. Writes CVPR_paper/figures/stream_grid_{linear,log}.pdf.
usage: stream_figs.py [--suffix stream50]
"""
import argparse, glob, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

R = 'results/things_eeg/cvpr/final'
SOA = 0.2
REPS = [5, 10, 20, 40]
# lightness-matched one-hue ramps (OKLCH L .76/.62/.48/.34), validated: each ordinal; red vs blue at every q tint
# CVD dE >= 15. Grey vs the extreme tints is below the normal-vision floor: zero-shot is drawn thin as the relief.
TINT = {'red': ['#fd8e86', '#e34948', '#b0081d', '#6d010d'], 'blue': ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'],
        'grey': ['#b2b2b2', '#878787', '#5d5d5d', '#373737']}
STYLE = {'ATM-128': '-', 'SAMGA': ':'}
# (label, run dir glob, kind, method, hue, encoder)
ARMS = [('SAGE, ATM-128', 'atm128_s33*', 'sage', 'full1_3', 'red', 'ATM-128'),
        ('SCORE, ATM-128', 'atm128_s33*', 'score', 'score', 'blue', 'ATM-128'),
        ('SCORE (original)', 'score_orig_s33*', 'score', 'score', 'blue', 'SAMGA'),
        ('Zero-shot, ATM-128', 'atm128_s33*', 'score', 'plain', 'grey', 'ATM-128'),
        ('Zero-shot, SAMGA', 'score_orig_s33*', 'score', 'plain', 'grey', 'SAMGA')]
COLS = [('', 'Buffer'), ('@new', 'Prequential, at arrival'), ('@prev', 'Prequential, strict-causal')]


def curves(files, method):
    d = pd.concat([pd.read_csv(f) for f in files])
    out = {}
    for suf, _ in COLS:
        x = d[d.method == method + suf].sort_values('t')
        if x.empty:
            continue
        if suf:  # cumulative mean over chunks, per stream
            x = x.assign(top1=x.groupby(['subject', 'menu', 'reps', 'seed']).top1.transform(lambda s: s.expanding().mean()))
        out[suf] = x.groupby(['menu', 'reps', 't']).top1.mean().reset_index()
    return out, d.subject.nunique()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suffix', default='stream50')
    a = p.parse_args()
    data = []
    for label, runs, kind, method, hue, enc in ARMS:
        files = glob.glob(f'{R}/{runs}/{a.suffix}_{kind}/part_*/stream_rows.csv')
        if files:
            c, n = curves(files, method)
            data.append((label, hue, enc, c))
            print(f'{label:20s} {len(files)} parts, {n} subjects')
    menus = sorted(set().union(*[set(c[''].menu) for *_, c in data]))
    for scale in ('linear', 'log'):
        fig, axs = plt.subplots(len(menus), len(COLS), figsize=(7.0, 1.75 * len(menus)), sharex='row', sharey='row', squeeze=False)
        for i, K in enumerate(menus):
            for j, (suf, title) in enumerate(COLS):
                ax = axs[i, j]
                for label, hue, enc, c in data:
                    if suf not in c:
                        continue
                    for k, q in enumerate(REPS):
                        h = c[suf][(c[suf].menu == K) & (c[suf].reps == q)]
                        ax.plot(h.t * q * SOA, h.top1, color=TINT[hue][k], ls=STYLE[enc], lw=0.8 if hue == 'grey' else 1.4)
                ax.set_xscale(scale)
                if scale == 'log':
                    x0 = min((c[''][c[''].menu == K].t * c[''][c[''].menu == K].reps).min() * SOA for *_, c in data)  # same for every q
                    ax.set_xticks([t for t in (3, 10, 30, 100, 300, 1000, 3000) if x0 * 0.9 <= t <= K * 80 * SOA * 1.05])
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(axis='y', lw=0.4, color='#dddddd')
                if i == 0:
                    ax.set_title(title, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f'K = {K}\nTop-1 (%)')
                if i == len(menus) - 1:
                    ax.set_xlabel('Recording time (s)')
        hand = [Line2D([], [], color=TINT[h][2], ls=STYLE[e], lw=0.8 if h == 'grey' else 1.4, label=l) for l, h, e, _ in data]
        hand += [Line2D([], [], color=TINT['red'][k], lw=1.4, label=f'q = {q}' + (' (tint, every method)' if k == 0 else '')) for k, q in enumerate(REPS)]
        fig.legend(handles=hand, loc='upper center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3)
        os.makedirs('CVPR_paper/figures', exist_ok=True)
        fig.savefig(f'CVPR_paper/figures/stream_grid_{scale}.pdf')
        fig.savefig(f'CVPR_paper/figures/stream_grid_{scale}.png', dpi=160)
    print('wrote CVPR_paper/figures/stream_grid_{linear,log}.{pdf,png}')


if __name__ == '__main__':
    main()
