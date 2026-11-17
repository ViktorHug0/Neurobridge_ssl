"""Figures and table rows of the incremental setting, from results/things_eeg/cvpr/inc/*.
  figures/incremental.pdf   top-1 on the buffer vs stimulus time, uniform / skewed / partial streams (q = 20)
  figures/stream_full_<stream>.pdf   appendix: every menu size K x number of repetitions q
The figures show the mean over the three training seeds, with a band of +-1 standard deviation over the seeds.
and prints the LaTeX rows of the main, ablation and encoder tables. Accuracy at a given stimulus time is interpolated
linearly in log-time between the two nearest refits.
SCORE is reported twice (user, 2026-11-10): as its own pipeline (its encoder, SAMGA + recovery-aware training, one seed;
method own_score) and with its deployment step on ATM (method score). Both use the constants of our reproduction
(k = 5, rho = 0.05; 52.35 in batch on its own encoder). Every other row is on ATM.
GALA = prior of 2d observations at the zero-shot decoder (user, 2026-11-10 23:10; gala.fit default).
Sources: v7/ (main text, q = 20), v5/ (appendix grid: full = uniform, skewed and the earlier half-coverage stream;
skew25 = the paper's partial stream).
usage: make_inc_figures.py
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = 'results/things_eeg/cvpr/inc'
OUT = 'CVPR_paper/figures'
SOA = 0.2 / 60  # minutes of stimulus per repetition
# the paper's partial stream is skew25 (user, 2026-11-11): a quarter of the gallery never queried, skewed counts on the rest.
# The earlier partial streams ('partial': half never queried, 'partial25': a quarter, both with equal counts) are still drawn.
STREAMS = {'uniform': 'Uniform', 'skew': 'Skewed', 'skew25': 'Partial'}
# name, colour, line style; the figures show SHOW only (readability, user 2026-11-11), the tables every method
SHOW = ['plain', 'sattc', 'score', 'gala']
STYLE = {'plain': ('Zero-shot', '#8a8a8a', '-', True), 'csls': ('CSLS', '#b0a070', '-', False),
         'adabn': ('AdaBN', '#c08fd0', '-', True), 'tent': ('Tent', '#c08fd0', ':', False),
         'ea': ('Euclidean re-centering', '#a0522d', '-', False), 'ra': ('Riemannian re-centering', '#a0522d', ':', False),
         'sattc': ('SATTC', '#e0a020', '-', True), 'transclip': ('TransCLIP', '#1baf7a', '-', True),
         'em_dirichlet': ('EM-Dirichlet', '#9b59b6', '-', True), 'hungarian': ('Hungarian assignment', '#555555', '--', True),
         'own_score': ('SCORE (own encoder)', '#eb6834', '--', True), 'score': ('SCORE (ATM)', '#eb6834', '-', True),
         'gala_noshrink': ('w/o prior', '#2a78d6', ':', False), 'gala_damp': ('post-hoc damping',), 'gala_d': ('prior of $d$ observations',), 'gala': ('GALA', '#2a78d6', '-', True),
         'own_plain': ('Zero-shot',), 'own_gala': ('GALA',), 'gala_nowhiten': ('w/o whitening',), 'gala_nocsls': ('w/o CSLS',),
         'gala_1step': ('single iteration',), 'gala_norot': ('w/o rotation',), 'gala_softmax': ('w/o Sinkhorn',)}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})


def load(pattern, keep=None, prefix='', rename=None):
    fs = glob.glob(f'{R}/{pattern}')
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(run=f.split('/')[-1].split('_s33')[0], seed=f.split('_s33')[1][:2]) for f in fs])
    d['method'] = d.method.replace({'gala_noprior': 'gala_noshrink', **(rename or {})})
    return d[d.method.isin(keep)] if keep else d.assign(method=prefix + d.method)


def curves(*ds):
    """mean top-1 per (scenario, q, method, t): over orders, then subjects, then seeds."""
    d = pd.concat([x for x in ds if x is not None])
    d = d.groupby(['seed', 'subject', 'scenario', 'q', 'method', 't']).top1.mean().reset_index()
    return d.groupby(['scenario', 'q', 'method', 't']).top1.mean()


def seed_std(d, keys):
    """std over the training seeds of the per-seed mean curve (over subjects and orders): the bands of the figures."""
    return d.groupby(['seed'] + keys).top1.mean().groupby(keys).std()


def draw(ax, h, sd, x, m, lw):
    ax.fill_between(x, h.values - sd.values, h.values + sd.values, color=STYLE[m][1], alpha=0.22, lw=0, zorder=1)
    ax.plot(x, h.values, color=STYLE[m][1], lw=lw, label=STYLE[m][0], zorder=3 if m == 'gala' else 2)


def at(c, sc, q, m, minutes):
    if (sc, q, m) not in c.index.droplevel('t'):
        return np.nan
    h = c.loc[(sc, q, m)]
    t = minutes / (q * SOA)
    return np.interp(np.log(min(t, h.index.max())), np.log(h.index.values), h.values)


def zoom(curves_, pad=0.04):
    """y-limits that just contain the given (curve, band half-width) pairs."""
    lo, hi = min((h - s).min() for h, s in curves_), max((h + s).max() for h, s in curves_)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def table(c, q, methods, cols):
    for m in methods:
        if (cols[0][0], q, m) in c.index.droplevel('t'):
            print(f'{STYLE.get(m, (m,))[0]} & ' + ' & '.join(f'{at(c, sc, q, m, mn):.1f}' for sc, mn in cols) + r' \\')


q = 20  # main text (user, 2026-11-11): 20 repetitions per query, so that the three streams last at least 20 minutes
# v7 = every main-text run at q = 20 under today's method names (Slurm job 73807). The q = 10 runs it supersedes are in
# inc/main, v2/main, v3/{main,main2d,ownfast,enc,encscore}, v4/{abl,enc}, v5/encsattc, v6 and inc/model.
ATM = [load('v7/main/*.csv')]
c = curves(*ATM, load('v7/model3/*.csv'), load('v7/own/*.csv', prefix='own_'))
end = c.loc[('uniform', q, 'plain')].index.max() * q * SOA
cols = [(sc, mn) for sc in STREAMS for mn in (1, 5, 20) + ((end,) if sc == 'uniform' else ())]
print('columns:', [(sc, round(mn, 1)) for sc, mn in cols])
table(c, q, ['plain', 'csls', 'adabn', 'tent', 'ea', 'ra', 'sattc', 'transclip', 'em_dirichlet', 'hungarian', 'own_score', 'score', 'gala'], cols)
print("--- on SCORE's encoder: zero-shot / SCORE")
table(c, q, ['own_plain', 'own_score'], cols)

sd = seed_std(pd.concat(ATM), ['scenario', 'q', 'method', 't'])
# incremental.pdf: the four methods of the paper; incremental_6methods.pdf: the same with TransCLIP and EM-Dirichlet
for show, name_ in ((SHOW, 'incremental'), (['plain', 'sattc', 'em_dirichlet', 'transclip', 'score', 'gala'], 'incremental_6methods')):
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True)
    for ax, (sc, name) in zip(axs, STREAMS.items()):
        for m in show:
            h = c.loc[(sc, q, m)]
            draw(ax, h, sd.loc[(sc, q, m)], h.index * q * SOA, m, 1.9 if m == 'gala' else 1.2)
        ax.set(xlabel='Stimulus time (min)', title=f'{name} stream', ylim=zoom([(c.loc[(s_, q, m)], sd.loc[(s_, q, m)]) for s_ in STREAMS for m in show]), xlim=(0, None))
        ax.grid(axis='y', lw=0.4, color='#dddddd')
    axs[0].set_ylabel('Top-1 on the buffer (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc='lower center', ncol=len(show), fontsize=7.5, handlelength=2.0, columnspacing=1.3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.09, 1, 1))
    fig.savefig(f'{OUT}/{name_}.pdf')
    fig.savefig(f'{OUT}/{name_}.png', dpi=200)

print('--- ablation: 1 / 20 min per stream')
table(c, q, ['gala', 'gala_d', 'gala_noshrink', 'gala_damp', 'gala_nowhiten', 'gala_nocsls', 'gala_softmax', 'gala_1step', 'gala_norot', 'plain'],
      [(sc, mn) for sc in STREAMS for mn in (1, 20)])

enc = [load('v7/enc3/*.csv'), *ATM]  # v7/enc = the same with one stream order
if enc[0] is not None:
    print('--- encoders (uniform stream): 1 / 5 / 10 / 20 min / end')
    d = pd.concat([x for x in enc if x is not None])
    d = d[(d.scenario == 'uniform') & (d.q == q)].groupby(['run', 'method', 't']).top1.mean()
    for mn in (1, 5, 10, 20, 'end'):
        rows = {}
        for (run, m), h in d.groupby(level=[0, 1]):
            h = h.droplevel([0, 1])
            rows.setdefault(m, {})[run] = h.iloc[-1] if mn == 'end' else np.interp(np.log(mn / (q * SOA)), np.log(h.index.values), h.values)
        print(mn, '\n', pd.DataFrame(rows).T.round(1).to_string())

# appendix: every menu size K x number of repetitions q, one figure per stream (ATM, 3 seeds, 3 orders, 15 refits)
full = load('v5/full/atm*.csv')  # 3 seeds x 3 stream orders (Slurm job 72922); v3/full, v3/full2d = the earlier 1-seed, 1-order run
full = pd.concat([x for x in (full, load('v5/partial25/*.csv'), load('v5/skew25/*.csv')) if x is not None])
if full is not None:
    d = pd.concat([full, *[load(f, ['own_score'], rename={'score': 'own_score'}) for f in ('v5/full/score30s2*.csv', 'v5/skew25own/*.csv')]])
    print('--- full grid, seeds x subjects per K:', full[full.scenario == 'uniform'].groupby(['K', 'seed']).subject.nunique().groupby('K').sum().to_dict())
    cf = d.groupby(['seed', 'subject', 'K', 'scenario', 'q', 'method', 't']).top1.mean().groupby(['K', 'scenario', 'q', 'method', 't']).mean()
    sf = seed_std(full, ['K', 'scenario', 'q', 'method', 't'])
    Ks, qs = sorted(full.K.unique()), sorted(full.q.unique())
    M = ['plain', 'csls', 'sattc', 'transclip', 'em_dirichlet', 'hungarian', 'own_score', 'score', 'gala']  # for the printed summary
    def grid_figure(cf, sf, sc, name):
        fig, axs = plt.subplots(len(Ks), len(qs), figsize=(6.9, 1.55 * len(Ks) + 0.4), sharey='row', squeeze=False)
        for i, K in enumerate(Ks):
            for j, qq in enumerate(qs):
                ax = axs[i, j]
                for m in SHOW:
                    h = cf.loc[(K, sc, qq, m)]
                    draw(ax, h, sf.loc[(K, sc, qq, m)], h.index * qq * SOA, m, 1.6 if m == 'gala' else 1.0)
                ax.set(ylim=zoom([(cf.loc[(K, sc, q_, m)], sf.loc[(K, sc, q_, m)]) for q_ in qs for m in SHOW]), xlim=(0, None))  # y zoomed per row
                ax.tick_params(labelsize=6)
                ax.grid(axis='y', lw=0.3, color='#e4e4e4')
                if i == 0:
                    ax.set_title(f'$q$ = {qq}', fontsize=8)
                if j == 0:
                    ax.set_ylabel(f'$K$ = {K}\nTop-1 (%)', fontsize=7)
                if i == len(Ks) - 1:
                    ax.set_xlabel('Stimulus time (min)', fontsize=7)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc='lower center', ncol=4, fontsize=7.5, handlelength=2.2, columnspacing=1.6, bbox_to_anchor=(0.5, 0))
        fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.4, rect=(0, 0.03, 1, 1))
        fig.savefig(f'{OUT}/{name}_{sc}.pdf')
        fig.savefig(f'{OUT}/{name}_{sc}.png', dpi=150)
    for sc in [*STREAMS, 'partial', 'partial25']:
        grid_figure(cf, sf, sc, 'stream_full')
    # causal protocol (inc_causal.py, a refit after every query): every query decoded by the calibration fitted on the
    # queries before it; a point = the accuracy on the queries of a window of 1/15 of the stream
    # GALA = gala_sw_freq1 (user, 2026-11-16: column masses estimated from the buffer, one iteration per update from the
    # previous rotation, stream-selected settings; inc/causal2_freq, main setting in inc/causal2/*_abl3_*); SCORE with
    # stream-selected settings (inc/causal2); zero-shot and SATTC from inc/causal1. Rows = windows of N / 100 queries
    cau = pd.concat([load('causal1/*.csv', keep=['plain', 'sattc']), load('causal2/*.csv', keep=['score', 'gala'], rename={'gala': '-', 'gala_sw_freq1': 'gala'}),
                     load('causal2_freq/*.csv', keep=['gala'], rename={'gala_sw_freq1': 'gala'})])
    if cau is not None:
        def windows(ts, frac=1 / 15):
            """refit sizes of one stream -> centre (in queries) of the window of each: consecutive refits are merged until
            the window holds at least frac of the stream (the first refits hold one or two queries each)."""
            wins, start, group = [], 0, []
            for t in ts:
                group.append(t)
                if t - start >= frac * ts[-1]:
                    wins.append((start, group))
                    start, group = t, []
            wins[-1][1].extend(group)
            return {t: (s0 + g[-1]) / 2 for s0, g in wins for t in g}
        ck = ['K', 'scenario', 'q', 'method', 't']
        w = {k: windows(sorted(v.unique())) for k, v in cau.groupby(['K', 'scenario', 'q']).t}
        cau['t'] = [w[K, sc, qq][t] for K, sc, qq, t in zip(cau.K, cau.scenario, cau.q, cau.t)]
        cau = cau.assign(hits=cau.top1 * cau.n).groupby(['seed', 'subject', 'order'] + ck)[['hits', 'n']].sum().reset_index()
        cau['top1'] = cau.hits / cau.n  # accuracy on the queries of the window
        cc = cau.groupby(['seed', 'subject'] + ck).top1.mean().groupby(ck).mean()
        cs = seed_std(cau, ck)
        for sc in STREAMS:
            grid_figure(cc, cs, sc, 'final_stream_full')
        # main-text figure: the main setting (K = 200, q = 20)
        fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True)
        for ax, (sc, name) in zip(axs, STREAMS.items()):
            for m in SHOW:
                h = cc.loc[(200, sc, q, m)]
                draw(ax, h, cs.loc[(200, sc, q, m)], h.index * q * SOA, m, 1.9 if m == 'gala' else 1.2)
            ax.set(xlabel='Stimulus time (min)', title=f'{name} stream', ylim=zoom([(cc.loc[(200, s_, q, m)], cs.loc[(200, s_, q, m)]) for s_ in STREAMS for m in SHOW]), xlim=(0, None))
            ax.grid(axis='y', lw=0.4, color='#dddddd')
        axs[0].set_ylabel('Top-1 on incoming queries (%)')
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc='lower center', ncol=len(SHOW), fontsize=7.5, handlelength=2.0, columnspacing=1.3, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.09, 1, 1))
        fig.savefig(f'{OUT}/final_incremental.pdf')
        fig.savefig(f'{OUT}/final_incremental.png', dpi=200)
    # numbers for the commentary: end of stream and one fifth of the stream, per cell
    e = cf.groupby(level=['K', 'scenario', 'q', 'method']).last().unstack('method').reindex(columns=M).round(1)
    print(e.to_string())
    # abstract and appendix summary: GALA minus the best other method, per configuration (stream x K x q), at given times
    comp = [m for m in M if m not in ('gala', 'plain')]
    def margin(K, sc, qq, minutes):
        """(GALA - best competitor, GALA - zero-shot) at this stimulus time, or None when the stream is shorter."""
        v = {}
        for m in M:
            h = cf.loc[(K, sc, qq, m)]
            if minutes is None:
                v[m] = h.iloc[-1]
            elif minutes > h.index.max() * qq * SOA:
                return None
            else:
                v[m] = np.interp(np.log(minutes / (qq * SOA)), np.log(h.index.values), h.values)
        return v['gala'] - max(v[m] for m in comp), v['gala'] - v['plain'], max(comp, key=v.get)
    print('--- GALA minus best competitor over the stream x K x q configurations (mean; configurations that last that long)')
    for mn in (1, 5, 10, 20, None):
        r = [(sc, K, qq, margin(K, sc, qq, mn)) for sc in STREAMS for K in Ks for qq in qs]
        r = [(sc, K, qq, *x) for sc, K, qq, x in r if x is not None]
        g = np.array([x[3] for x in r])
        print(f"{'end' if mn is None else mn} min: n = {len(r)}, mean {g.mean():.1f}, median {np.median(g):.1f}, GALA best in {(g > 0).sum()}, "
              f"mean over zero-shot {np.mean([x[4] for x in r]):.1f}, per stream "
              + ', '.join(f"{sc} {np.mean([x[3] for x in r if x[0] == sc]):.1f} (n={sum(x[0] == sc for x in r)})" for sc in STREAMS)
              + ', best competitor: ' + str(pd.Series([x[5] for x in r]).value_counts().to_dict()))

