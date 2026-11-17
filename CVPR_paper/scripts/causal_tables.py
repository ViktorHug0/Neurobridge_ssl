"""Numbers of the causal protocol (inc_causal.py: every query decoded by the calibration fitted on the queries before
it, one update per query) for the paper, printed as LaTeX rows.

GALA = gala_sw_freq1 (user, 2026-11-16): Sinkhorn towards column masses estimated from the buffer, one iteration per
update from the previous rotation, settings selected on the streams of the other subjects (inc/causal2_freq; main
setting in inc/causal2/*_abl3_*). SCORE with ATM = stream-selected settings (inc/causal2). Zero-shot, CSLS, SATTC,
Hungarian, TransCLIP, EM-Dirichlet: inc/causal1. AdaBN / Tent: inc/causal1_model (updated every 20 queries; their
1-minute value is the first 20 queries). SCORE with its own encoder: inc/causal1_own.
Earlier methods, for reference: gala (inc/causal1: balanced Sinkhorn, 16 iterations from scratch, batch-selected
settings) and gala_sw (inc/causal2: no Sinkhorn).

1. 60 settings (stream x gallery size x repetitions), accuracy on the last 30 % of each stream, pooled: lead over the
   best of CSLS / SATTC / Hungarian / SCORE with ATM.
2. Main setting (K = 200, q = 20): accuracy at 1 / 5 / 20 / 53 minutes (queries within +-30 % of 15 m; 53 min = last 10 %).
3. Ablation (main setting). 4. Encoders (uniform stream, 3 orders, one seed; ATM column from the main setting).
usage: causal_tables.py
"""
import glob

import pandas as pd

R = 'results/things_eeg/cvpr/inc/'
KEY = ['file', 'subject', 'K', 'q', 'scenario', 'order']
STREAMS = ('uniform', 'skew', 'skew25')
pd.set_option('display.width', 250)


def load(pattern, field=2):  # field of the file name kept as column 'file': 2 = training seed, 0 = encoder
    return pd.concat([pd.read_csv(f).assign(file=f.split('/')[-1].split('_')[field]) for f in glob.glob(R + pattern)])


def pooled(x, by):
    g = x.assign(h=x.top1 * x.n).groupby(by)[['h', 'n']].sum()
    return g.h / g.n


def at(x, m):
    """rows of one stream type (q = 20) at minute m"""
    if m == 53:
        return x[x.t > 0.9 * x.t.max()]
    w = x[(x.t >= 0.7 * 15 * m) & (x.t <= 1.3 * 15 * m)]
    return w if len(w) else x[x.t == x.t.min()]  # AdaBN / Tent at 1 minute: their first window (queries 1-20)


def row(name, v, bold=()):
    print(f'{name} & ' + ' & '.join((r'\textbf{%.1f}' if i in bold else '%.1f') % x for i, x in enumerate(v)) + r' \\')


c1, c2 = load('causal1/atm128_vc_s*.csv'), pd.concat([load('causal2/atm128_vc_s*.csv'), load('causal2_freq/atm128_vc_s*.csv')])

# 1. the 60 settings
def end(x, methods):
    x = x[x.method.isin(methods)]
    x = x[x.t > 0.7 * x.groupby(['scenario', 'K', 'q']).t.transform('max')]
    n = x.drop_duplicates(KEY + ['method']).groupby(['scenario', 'K', 'q', 'method']).size().unstack('method')
    return pooled(x, ['scenario', 'K', 'q', 'method']).unstack('method'), n

old, _ = end(c1, ['plain', 'csls', 'sattc', 'hungarian', 'score', 'gala'])
new, n = end(c2, ['score', 'gala_sw', 'gala_sw_freq1'])
print('streams per setting (min, max):', n.min().to_dict(), n.max().to_dict())
t = old.join(new.rename(columns={'score': 'score_sel'}))
base = t[['csls', 'sattc', 'hungarian']].max(axis=1)
for m, s in (('gala', 'score'), ('gala_sw', 'score_sel'), ('gala_sw_freq1', 'score_sel')):
    t['lead_' + m] = t[m] - pd.concat([base, t[s]], axis=1).max(axis=1)
    L = t['lead_' + m]
    print(f'{m:14s} ahead in {(L > 0).sum()} of {L.notna().sum()}, mean lead {L.mean():.1f} | per stream (ahead, mean lead, mean accuracy)',
          {sc: (int((L.loc[sc] > 0).sum()), round(float(L.loc[sc].mean()), 1), round(float(t[m].loc[sc].mean()), 1)) for sc in STREAMS})
L = t.lead_gala_sw_freq1
print('GALA lead per stream (min, median, max):', {sc: tuple(L.loc[sc].agg(['min', 'median', 'max']).round(1)) for sc in STREAMS},
      '| over zero-shot, mean:', round(float((t.gala_sw_freq1 - t.plain).mean()), 1),
      '| second best:', t[['csls', 'sattc', 'hungarian', 'score_sel']].idxmax(axis=1).value_counts().to_dict())
print('GALA lead by gallery size:', L.groupby('K').mean().round(1).to_dict(), '| by repetitions:', L.groupby('q').mean().round(1).to_dict())
print(t.round(1).to_string())

# 2. main setting
COLS = [(sc, m) for sc in STREAMS for m in (1, 5, 20) + ((53,) if sc == 'uniform' else ())]
def minutes(x):
    x = x[(x.K == 200) & (x.q == 20)]
    return pd.DataFrame({(sc, m): pooled(at(x[x.scenario == sc], m), 'method') for sc, m in COLS})

own = minutes(load('causal1_own/*.csv')).rename(index={'score': 'own_score', 'plain': 'own_plain'})
T = pd.concat([minutes(c1).drop(index=['score', 'gala_freq1', 'gala_warm1'], errors='ignore'),  # the two = tests of 2026-11-16 with batch-selected settings
               minutes(c2), minutes(load('causal1_model/*.csv')), own])
print(T.round(1).to_string())
MAIN = [('Zero-shot', 'plain'), ('CSLS', 'csls'), (r'AdaBN~\cite{li2018adabn}', 'adabn'), (r'Tent~\cite{wang2021tent}', 'tent'),
        (r'SATTC~\cite{huang2026sattc}', 'sattc'), (r'TransCLIP~\cite{zanella2024transclip}', 'transclip'),
        (r'EM-Dirichlet~\cite{martin2024transductive}', 'em_dirichlet'), ('Hungarian assignment', 'hungarian'),
        (r'SCORE~\cite{cui2026score}', 'own_score'), ('SCORE with ATM', 'score'), ('GALA', 'gala_sw_freq1')]
best = T.loc[[m for _, m in MAIN]].round(1).max()
print('--- main table')
for name, m in MAIN:
    row(name, T.loc[m], [i for i, c in enumerate(COLS) if round(T.loc[m, c], 1) == best[c]])

# 3. ablation: 1 / 20 minutes of each stream, and the end of the uniform stream
print('--- ablation: 1 / 20 min per stream, then 53 min uniform')
A = [(sc, m) for sc in STREAMS for m in (1, 20)] + [('uniform', 53)]
for name, m in (('GALA', 'gala_sw_freq1'), ('w/o whitening', 'gala_swf_nowhiten'), ('w/o CSLS', 'gala_swf_nocsls'), ('uniform column masses', 'gala_sw_sinkhorn'),
                ('w/o column constraint', 'gala_sw'), ('w/o rotation', 'gala_s_norot'), ('w/o prior', 'gala_swf_noprior'),
                ('16 iterations from scratch', 'gala_freq1'), ('Zero-shot', 'plain')):
    if m in T.index:
        row(name, T.loc[m, A])

# 4. encoders (q = 20): uniform stream from causal1_enc (zero-shot, SATTC) + causal2_enc (SCORE, GALA); skewed and
# partial streams from causal2_enc_sp (all four methods); ATM column from the main setting
ORDER = ['atm', 'tsconv', 'eegnet', 'eegproject', 'conformer', 'labram']
sp = glob.glob(R + 'causal2_enc_sp/*.csv')
for sc, E in (('uniform', (1, 5, 10, 20, 53)), ('skew', (1, 5, 10, 20)), ('skew25', (1, 5, 10, 20))):
    if sc == 'uniform':
        enc = pd.concat([load('causal1_enc/*.csv', 0).query("method in ['plain', 'sattc']"), load('causal2_enc/*.csv', 0)])
    elif sp:
        enc = load('causal2_enc_sp/*.csv', 0)
        enc = enc[enc.scenario == sc]
    else:
        continue
    enc = enc.rename(columns={'file': 'encoder'})
    print(sc, 'encoder streams:', enc.drop_duplicates(['encoder', 'method', 'subject', 'order']).groupby(['encoder', 'method']).size().unstack().to_dict('index'))
    atm = pd.concat([c1[c1.method.isin(['plain', 'sattc'])], c2])
    atm = atm[(atm.K == 200) & (atm.q == 20) & (atm.scenario == sc)].assign(encoder='atm')
    enc = pd.concat([enc, atm])
    tab = pd.DataFrame({m: pooled(at(enc, m), ['encoder', 'method']) for m in E}).round(1)
    print(f'--- encoder table, {sc}')
    for m in E:
        v = tab[m].unstack('encoder').reindex(columns=ORDER).loc[['plain', 'sattc', 'score', 'gala_sw_freq1']]
        for name, meth in (('Zero-shot', 'plain'), ('SATTC', 'sattc'), ('SCORE', 'score'), ('GALA', 'gala_sw_freq1')):
            print((f'{m} & ' if meth == 'plain' else ' & ') + f'{name} & ' + ' & '.join(
                (r'\textbf{%.1f}' if x == v[e].max() else '%.1f') % x for e, x in v.loc[meth].items()) + r' \\')
