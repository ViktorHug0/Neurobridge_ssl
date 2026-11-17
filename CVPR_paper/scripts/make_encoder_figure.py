"""Paper Fig. 3: which encoders benefit from calibration. Writes CVPR_paper/figures/encoders.pdf.
The 6 encoders of Table 2 (ValCon, 10 LOSO folds, cross-fold parameters) and ATM trained against 5 other image targets.
y = fraction of the zero-shot errors fixed by GALA. (a) x = zero-shot top-1, (b) x = RSA.
2026-11-16: GALA = the final method (estimated column masses) without the prior, settings selected on the batch evaluation of
the other folds (appendix_c.py -> analysis/encoder_gain_final.csv); before: uniform masses (encoder_gain_diagnostics*_vc.csv).
Least-squares line with its 95% confidence band over all 11 points; rho = Spearman over all 11 points.
The five ATM image-target variants (open diamonds) were added on 2026-10-26 (user: merge into panel b).
usage: make_encoder_figure.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

A = 'results/things_eeg/cvpr/analysis'
BLUE, GREY, BAND = '#2a78d6', '#8a8a8a', '#cfe0f6'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
ENC = {'atm128': 'ATM', 'tsconv': 'TSConv', 'eegnet': 'EEGNet', 'eegproject': 'EEGProject', 'conformer': 'Conformer',
       'labram': 'LaBraM'}

g = pd.read_csv(f'{A}/encoder_gain_final.csv').rename(columns={'m_gala': 'm_sage'})
e = g[~g.arm.isin([f'{k}_vc_s3300' for k in ('atm128_iv20', 'atm128_iv44', 'atm128_dino40', 'atm128_cliph', 'atm128_bigg27')])].copy()  # ValCon checkpoints
e['enc'] = e.arm.str.replace(r'_(vc|valcon)_s33\d\d', '', regex=True)
e = e.groupby('enc')[['m_plain', 'm_sage', 'rsa']].mean().loc[list(ENC)].rename(columns={'m_plain': 'plain', 'm_sage': 'sage'})
e['ff'] = 100 * (e.sage - e.plain) / (100 - e.plain)
# ATM-128 trained against other image targets (ValCon, seed 3300): final / near-final layers vs intermediate ones
TGT = {'atm128_iv20': 'InternViT L20', 'atm128_iv44': 'InternViT L44', 'atm128_dino40': 'DINOv2 L40',
       'atm128_cliph': 'CLIP-H final', 'atm128_bigg27': 'bigG L27'}
t = g[~g.index.isin(e.index)].copy()
t['enc'] = t.arm.str.replace(r'_vc_s33\d\d', '', regex=True)
t = t.groupby('enc')[['m_plain', 'm_sage', 'rsa']].mean().loc[list(TGT)].rename(columns={'m_plain': 'plain', 'm_sage': 'sage'})
t['ff'] = 100 * (t.sage - t.plain) / (100 - t.plain)
allp = pd.concat([e, t])
NAMES = {**ENC, **TGT}


def band(ax, x, y):
    """OLS line and 95% confidence band of the mean."""
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    res = y - (a + b * x)
    s = np.sqrt((res ** 2).sum() / (len(x) - 2))
    se = s * np.sqrt(1 / len(x) + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    q = student_t.ppf(0.975, len(x) - 2)
    ax.fill_between(xs, a + b * xs - q * se, a + b * xs + q * se, color=BAND, lw=0, zorder=1)
    ax.plot(xs, a + b * xs, color=BLUE, lw=1.0, alpha=0.6, zorder=2)


fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))
# label positions (data units), set by hand; every label has a leader line
POS = {'plain': {'atm128': (27.0, 84), 'atm128_iv20': (24.5, 74), 'atm128_bigg27': (32.6, 60), 'tsconv': (32.4, 34),
                 'conformer': (22.6, 60), 'eegnet': (16.0, 52), 'eegproject': (18.0, 31), 'labram': (25.5, 28),
                 'atm128_cliph': (12.3, 26), 'atm128_iv44': (16.0, 4), 'atm128_dino40': (21.0, 8)},
       'rsa': {'atm128': (0.648, 86), 'atm128_iv20': (0.632, 76), 'atm128_bigg27': (0.690, 54), 'tsconv': (0.610, 60),
               'conformer': (0.688, 40), 'eegnet': (0.602, 48), 'eegproject': (0.668, 27), 'labram': (0.612, 33),
               'atm128_cliph': (0.497, 27), 'atm128_iv44': (0.528, 4), 'atm128_dino40': (0.600, 8)}}
for ax, col, xlabel, tag in ((axs[0], 'plain', 'Zero-shot top-1 (%)', '(a)'), (axs[1], 'rsa', 'RSA between EEG and image embeddings', '(b)')):
    band(ax, allp[col].values, allp.ff.values)
    ax.scatter(e[col], e.ff, s=34, color=BLUE, zorder=4, label='EEG encoders')
    ax.scatter(t[col], t.ff, s=34, marker='D', facecolor='white', edgecolor=BLUE, lw=1.2, zorder=4, label='ATM, other image targets')
    r, p = spearmanr(allp[col], allp.ff)
    ax.set_title(f'{tag} $\\rho$ = {r:.2f} ($p$ = {p:.1g})', fontsize=9.5)
    ax.set(xlabel=xlabel, ylim=(0, 100))
    ax.set_xlim((11, 36) if col == 'plain' else (0.49, 0.71))
    ax.grid(axis='y', lw=0.4, color='#e4e4e4')
    for k in allp.index:
        ax.annotate(NAMES[k], (allp[col][k], allp.ff[k]), xytext=POS[col][k], fontsize=7,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color=GREY, shrinkA=0, shrinkB=3))
    print(tag, col, round(r, 3), p)
axs[0].set_ylabel('Zero-shot errors fixed (%)')
axs[1].legend(loc='upper left', fontsize=7, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig('CVPR_paper/figures/encoders.pdf')
fig.savefig('CVPR_paper/figures/encoders.png', dpi=200)
