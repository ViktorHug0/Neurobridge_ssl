"""Presentation figures for the subject-key work. Run from anywhere: .venv/bin/python scripts/subject_key/figures/make_figures.py [name ...]
Sources: results/things_eeg/mi/*.csv (MI_LOG E07, E09, E17), KEY_LOG tables, figures/fig_data.npz (fig_data.py)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, '../../../results/things_eeg/mi')
INK, MUTED, GRID, SOFT = '#1B2430', '#5F6B7A', '#E3E7EC', '#C3CAD3'
VIOLET, TEAL, AMBER = '#5946B0', '#12987f', '#d98f00'          # validated as a 3-slot categorical set (dataviz validator, light)
TINT = {'att': '#F1EFFA', 'conv': '#EAF5F3', 'head': '#FBF4E3'}  # same families as encoder_architectures.png
mpl.rcParams.update({'font.family': 'Inter', 'font.size': 13, 'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': SOFT,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'savefig.dpi': 220,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + '.png'), bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print('wrote', name + '.png')


def key_swap_architectures():
    """MI_LOG E09: seen subjects decoded with their own mean ERP, with another training subject's, and the held-out subject."""
    a = pd.read_csv(f'{MI}/e09_arch.csv')
    names = {'sqf': 'Squeezeformer', 'tsconv': 'TSConv', 'conformer': 'EEGConformer', 'eegproject': 'EEGProject', 'atm128': 'ATM',
             'eegnet': 'EEGNet', 'rmix': 'TokenREVE', 'labram': 'LaBraM (pretrained)'}
    seen = a[~a.held].groupby('arch')[['own', 'other']].mean()
    new = a[a.held].groupby('arch').own.mean()
    order = (seen.own - seen.other).loc[list(names)].sort_values().index[::-1]   # largest drop at the bottom of the axis -> plotted first
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for y, k in enumerate(order):
        o, w, n = seen.own[k], seen.other[k], new[k]
        ax.plot([w, o], [y, y], color=SOFT, lw=3, solid_capstyle='round', zorder=1)
        ax.scatter(n, y, s=95, facecolor='white', edgecolor=MUTED, lw=1.8, zorder=2)
        ax.scatter(w, y, s=120, color=VIOLET, edgecolor='white', lw=2, zorder=3)
        ax.scatter(o, y, s=120, color=INK, edgecolor='white', lw=2, zorder=3)
        ax.text(86, y, f'−{o - w:.0f}', va='center', ha='left', color=INK, fontsize=13, fontweight=600)
    ax.set_yticks(range(len(order)), [names[k] for k in order], color=INK, fontsize=13.5)
    ax.set_xlim(10, 84); ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlabel('Top-1 accuracy (%), 200-way'); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.text(86, len(order) - 0.25, 'cost of\nthe swap', va='bottom', ha='left', color=MUTED, fontsize=11)
    h = [plt.Line2D([], [], marker='o', ls='', ms=10, color=INK, mec='white', mew=2, label='seen subject, own mean ERP'),
         plt.Line2D([], [], marker='o', ls='', ms=10, color=VIOLET, mec='white', mew=2, label="seen subject, another subject's mean ERP"),
         plt.Line2D([], [], marker='o', ls='', ms=9, mfc='white', mec=MUTED, mew=1.8, label='new subject')]
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.46, 1.0), ncol=3, fontsize=12, handletextpad=0.3, columnspacing=1.6)
    save(fig, 'key_swap_architectures')


def key_use_by_stage():
    """MI_LOG E07 (TSConv) and E17 (ATM): seen subject under a wrong key, the right key's mean activation restored at one stage."""
    stages = [('embed', 'token\nembed.', 'att'), ('attention', 'self-\nattention', 'att'), ('norm1', 'add &\nnorm', 'att'), ('ffn+norm2', 'feed-\nforward', 'att'), ('enc_norm', 'layer\nnorm', 'att'),
              ('t_lin', 'temporal\nconv', 'conv'), ('t_act', 'pool, BN,\nELU', 'conv'), ('s_lin', 'spatial\nconv', 'conv'), ('s_act', 'BN,\nELU', 'conv'), ('bb', '1×1 conv,\nflatten', 'conv'),
              ('fc1', 'linear', 'head'), ('res', 'residual\nGELU block', 'head'), ('ln', 'layer\nnorm', 'head'), ('align', 'projector', 'head')]
    pos = {s: i for i, (s, _, _) in enumerate(stages)}
    data = {'TSConv': pd.read_csv(f'{MI}/e07_key_C_mix.csv').groupby('stage').acc.mean(), 'ATM': pd.read_csv(f'{MI}/e17_atm_chain_atm128.csv').groupby('stage').acc.mean()}
    ref = {'TSConv': (66.8, data['TSConv']['none']), 'ATM': (data['ATM']['(own key)'], data['ATM']['(wrong key)'])}
    notes = {'TSConv': {'res': 'residual block  −14'}, 'ATM': {'attention': 'attention  −9', 'ffn+norm2': 'feed-forward  −11', 't_act': '−3', 'res': 'residual block  −10'}}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, gridspec_kw={'hspace': 0.16})
    for ax, name in zip(axes, ('TSConv', 'ATM')):
        d = data[name]; own, wrong = ref[name]
        for fam in TINT:
            idx = [i for i, s in enumerate(stages) if s[2] == fam]
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=TINT[fam], lw=0, zorder=0)
        ax.axhline(own, color=INK, lw=0.9, zorder=1); ax.axhline(wrong, color=INK, lw=0.9, zorder=1)
        ax.text(13.6, own, ' own key', va='center', ha='left', fontsize=11.5, color=INK); ax.text(13.6, wrong, ' wrong key', va='center', ha='left', fontsize=11.5, color=INK)
        xs = [pos[s] for s, _, _ in stages if s in d.index]; ys = [d[s] for s, _, _ in stages if s in d.index]
        ax.plot(xs, ys, color=INK, lw=2, marker='o', ms=8, mec='white', mew=2, zorder=3, solid_joinstyle='round', clip_on=False)
        for s, lab in notes[name].items():
            ax.annotate(lab, (pos[s], d[s]), xytext=(8, 12), textcoords='offset points', fontsize=12, fontweight=600, color=INK, ha='left')
        ax.set_ylim(wrong - 6, own + 7); ax.set_xlim(-0.5, 13.5)
        ax.text(-0.35, own + 4.6, name, fontsize=16, fontweight=700, va='center')
        ax.set_ylabel('Top-1 (%)'); ax.tick_params(axis='x', length=0)
    axes[0].text(2, np.mean(ref['TSConv']), 'no electrode transformer', ha='center', va='center', color=MUTED, fontsize=12, style='italic')
    axes[1].set_xticks(range(len(stages)), [s[1] for s in stages], fontsize=10.5, color=INK)
    for x, lab, c in ((2, 'Electrode transformer', VIOLET), (7, 'Temporal–spatial convolution', '#1F7A70'), (12, 'Projection head', '#A8741A')):
        axes[0].text(x, 1.04, lab, transform=axes[0].get_xaxis_transform(), ha='center', va='bottom', fontsize=12.5, fontweight=700, color=c)
    fig.text(0.5, -0.03, 'Stage at which the mean activation of the right key is restored', ha='center', color=MUTED, fontsize=12.5)
    save(fig, 'key_use_by_stage')


def results_ladder():
    """ATM with three heads, 10 held-out subjects (KEY_LOG standing numbers, 3 seeds; simulated re-centring K48, seed 3300)."""
    rows = [('Plain model', 35.5), ('In-trial simulated re-centring', 48.5), ('Real re-centring', 52.6)]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    hgt = 0.5
    ax.set_xlim(0, 60); ax.set_ylim(-0.55, len(rows) - 0.3); fig.canvas.draw()
    box = ax.get_window_extent(); px_x, px_y = box.width / 60, box.height / (len(rows) + 0.25)
    for i, (name, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(FancyBboxPatch((0, y - hgt / 2), v, hgt, boxstyle=f'round,pad=0,rounding_size={5 / px_x * fig.dpi / 72}', fc=VIOLET, ec='none', mutation_aspect=px_x / px_y, zorder=2))
        ax.add_patch(plt.Rectangle((0, y - hgt / 2), v / 2, hgt, fc=VIOLET, ec='none', zorder=2, antialiased=False))
        ax.text(v - 0.9, y, f'{v:.1f}', va='center', ha='right', fontsize=14, fontweight=700, color='white', zorder=4)
        ax.text(-1.2, y, name, ha='right', va='center', fontsize=14, fontweight=600)
    ax.axvline(35.3, color=INK, lw=1, zorder=3, ymin=0, ymax=0.93)
    ax.text(35.3, len(rows) - 0.42, 'best published, 35.3', ha='center', va='bottom', fontsize=12, color=INK)
    ax.set_yticks([]); ax.set_xlabel('ATM, top-1 on a new subject (%), 200-way'); ax.spines['left'].set_color(SOFT)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, 'results_ladder')

CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
      'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
      'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
RAMP = mpl.colors.LinearSegmentedColormap.from_list('violet', ['#DAD4F4', '#8E7FDB', VIOLET, '#2B1C73'])   # one hue, light -> dark
_POS = []


def head(ax, w, label_min=0.10, scale=1.0):
    """Scalp seen from above; dot area and shade = share of the attention received by each electrode (w sums to ~1)."""
    if not _POS:
        import mne
        info = mne.create_info(CH, 250., 'eeg'); info.set_montage('standard_1005')
        from mne.channels.layout import _find_topomap_coords
        _POS.append(_find_topomap_coords(info, picks=np.arange(63)) / 0.095)
    p = _POS[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=SOFT, lw=1.3, zorder=1)
    ax.plot([-0.1, 0, 0.1], [0.995, 1.11, 0.995], color=SOFT, lw=1.3, zorder=1)
    for sx in (-1, 1):
        ax.plot(sx * np.array([1.0, 1.06, 1.07, 1.04, 1.0]), [0.18, 0.2, 0.0, -0.2, -0.18], color=SOFT, lw=1.3, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=7, color=SOFT, zorder=2, lw=0)
    o = np.argsort(w)
    area = scale * 3200 * np.clip(w - 0.02, 0, None)
    ax.scatter(p[o, 0], p[o, 1], s=area[o], c=w[o], cmap=RAMP, vmin=0, vmax=0.3, edgecolor='white', lw=1.2, zorder=3)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.22, 1.16); ax.set_aspect('equal')
    bb, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    upp = max(3.0 / (bb.width * fw * 72), 2.38 / (bb.height * fh * 72))   # data units per point
    for i in np.where(w >= label_min)[0]:
        off = np.sqrt(area[i] / np.pi) * upp + 0.07
        ax.text(p[i, 0] + (off if p[i, 0] >= -0.01 else -off), p[i, 1] - 0.03, f'{CH[i]}  {100 * w[i]:.0f}%', ha='left' if p[i, 0] >= -0.01 else 'right', va='center', fontsize=11, fontweight=600, zorder=4)
    ax.axis('off')


def attention_follows_mean_erp(a=6, b=3):
    """Frozen ATM (fold 1; both subjects seen in training). Trial = mean ERP + image-specific signal; the two parts are recombined across subjects."""
    d = np.load(f'{HERE}/fig_data.npz')
    recv = lambda M: np.asarray(M, dtype=float)[1:, 1:].mean(0)
    pairs = [tuple(p) for p in d['swap_pairs']]
    cell = {(a, a): d['A_own'][a - 1, 0], (b, b): d['A_own'][b - 1, 0], (a, b): d['swap_maps'][pairs.index((a, b))][0], (b, a): d['swap_maps'][pairs.index((b, a))][0]}  # (signal owner, ERP owner)
    fig = plt.figure(figsize=(14.5, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.55], wspace=0.05, hspace=0.02, left=0.13, right=0.99, top=0.86, bottom=0.1)
    for r, sig in enumerate((a, b)):
        for c, erp in enumerate((a, b)):
            ax = fig.add_subplot(gs[r, c]); head(ax, recv(cell[(sig, erp)]))
            if sig == erp:
                ax.text(-1.42, 1.02, 'real trial', fontsize=10.5, color=MUTED, style='italic', ha='left')
            if r == 0:
                ax.text(0, 1.36, f'mean ERP of subject {erp}', ha='center', fontsize=14, fontweight=700, color=INK)
        fig.text(0.115, 0.67 - 0.39 * r, f'image signal\nof subject {sig}', ha='right', va='center', fontsize=14, fontweight=700, color=INK)
    ax = fig.add_subplot(gs[:, 2])
    rows = [('Two images,\nsame subject', 'stat_same_key_other_image', True), ('Mean ERP alone\nvs a full trial', 'stat_key_alone', True),
            ('Recombined trial vs the\nowner of its mean ERP', 'stat_swap_vs_key_owner', True),
            ('Recombined trial vs the\nowner of its image signal', 'stat_swap_vs_content_owner', False), ('Two subjects,\nsame image', 'stat_same_image_other_subject', False)]
    rng = np.random.default_rng(0)
    for i, (lab, k, same) in enumerate(rows):
        y, v, c = len(rows) - 1 - i, d[k], VIOLET if same else MUTED
        ax.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=16, color=c, alpha=0.28, lw=0, zorder=2)
        ax.scatter(v.mean(), y, s=150, color=c, edgecolor='white', lw=2, zorder=3)
        ax.text(v.mean(), y + 0.3, f'{v.mean():.2f}', ha='center', va='bottom', fontsize=13, fontweight=600)
        ax.text(-0.04, y, lab, ha='right', va='center', fontsize=12.5, color=INK, linespacing=1.15)
    ax.set_xlim(0, 1.03); ax.set_ylim(-0.6, len(rows) - 0.3); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel('Correlation between two attention maps')
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.17, pos.y0 + 0.03, pos.width - 0.17, pos.height - 0.09])
    h = [plt.Line2D([], [], marker='o', ls='', ms=10, color=VIOLET, mec='white', mew=2, label='same mean ERP'), plt.Line2D([], [], marker='o', ls='', ms=10, color=MUTED, mec='white', mew=2, label='different mean ERP')]
    ax.legend(handles=h, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=12, handletextpad=0.2, columnspacing=1.4)
    fig.text(0.13, 0.03, 'Dot area: share of the attention each electrode receives (4 heads and 63 query tokens averaged), same test image in the four cells.', fontsize=11, color=MUTED, ha='left')
    save(fig, 'attention_follows_mean_erp')


def attention_by_subject():
    """Attention map computed from each subject's mean ERP alone (no image-specific signal). Subject 1 is the held-out subject of this model."""
    d = np.load(f'{HERE}/fig_data.npz')
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6), gridspec_kw={'wspace': 0.0, 'hspace': 0.12})
    for s, ax in enumerate(axes.flat):
        head(ax, d['A_key'][s, 1:, 1:].mean(0), label_min=0.12, scale=0.6)
        ax.text(0, 1.3, f'Subject {s + 1}' + (' (held out)' if s + 1 == int(d['fold']) else ''), ha='center', fontsize=13.5, fontweight=700)
    save(fig, 'attention_by_subject')


def key_steady_state(sub=6, ch='Oz'):
    """Mean ERPs (average of the 200 test queries, 80 repetitions each) of the ten subjects, and one query split into mean ERP + image-specific signal."""
    d = np.load(f'{HERE}/fig_data.npz'); mu, x = d['mu'], d['x']; t = np.arange(250) * 4; c = CH.index(ch)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, gridspec_kw={'hspace': 0.3})
    for ax in axes:
        for k in range(5):
            ax.axvspan(200 * k, 200 * k + 100, color='#F1F3F6', lw=0, zorder=0)
        ax.axhline(0, color=SOFT, lw=0.8, zorder=1); ax.set_xlim(0, 1000); ax.set_ylim(-1.3, 1.25); ax.set_ylabel(f'Amplitude at {ch} (a.u.)')
    for k, lab in enumerate(('decoded image', 'next image', '…', '…', '…')):
        axes[0].text(200 * k + 6, -1.22, lab, ha='left', va='bottom', fontsize=10.5, color=MUTED)
    for s in range(10):
        axes[0].plot(t, mu[s, c], color=SOFT, lw=1.3, zorder=2, label='other subjects' if s == 0 else None)
    for s, col in ((3, AMBER), (sub, VIOLET)):
        axes[0].plot(t, mu[s - 1, c], color=col, lw=2.2, zorder=3, label=f'subject {s}')
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[1:] + h[:1], l[1:] + l[:1], loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=11.5, handlelength=1.4, columnspacing=1.4)
    axes[0].set_title('Mean ERP of the ten subjects', loc='left', fontsize=14, fontweight=700, pad=10)
    tr = x[sub - 1, 0, c]
    for y, col, lab in ((tr, INK, 'query (80 repetitions)'), (mu[sub - 1, c], VIOLET, 'mean ERP'), (tr - mu[sub - 1, c], TEAL, 'difference: image-specific signal')):
        axes[1].plot(t, y, color=col, lw=2.2, label=lab, zorder=3)
    axes[1].legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=11.5, handlelength=1.4, columnspacing=1.4)
    axes[1].set_title(f'One query of subject {sub}', loc='left', fontsize=14, fontweight=700, pad=10)
    axes[1].set_xlabel('Time after the onset of the decoded image (ms)   ·   shaded: an image is on screen (5 Hz stream)')
    save(fig, 'key_steady_state')


def embedding_offset(seen=6):
    """Normalised embeddings of the 200 test queries and of their images (fold 1, subject 1 held out). x: unit vector along the new subject's offset; y: first PC of the centred clouds."""
    d = np.load(f'{HERE}/fig_data.npz'); E, g, anc = d['E_own'], d['g'], d['anchor']; new, old = E[int(d['fold']) - 1], E[seen - 1]
    off = new.mean(0) - g.mean(0); off /= np.linalg.norm(off)
    Xc = np.concatenate([g - g.mean(0), new - new.mean(0), old - old.mean(0)]); Xc -= np.outer(Xc @ off, off)
    pc = np.linalg.svd(Xc, full_matrices=False)[2][0]
    acc = lambda e: 100 * ((e @ g.T).argmax(1) == np.arange(200)).mean()
    rec = new - new.mean(0) + anc
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True, sharey=True, gridspec_kw={'wspace': 0.06})
    for ax, sets, title in ((axes[0], ((g, INK, 'images'), (old, TEAL, f'seen subject · {acc(old):.0f} %'), (new, VIOLET, f'new subject · {acc(new):.0f} %')), 'Model output'),
                            (axes[1], ((g, INK, 'images'), (rec, VIOLET, f'new subject, re-centred · {acc(rec):.0f} %')), 'After re-centring')):
        for Z, col, lab in sets:
            ax.scatter(Z @ off, Z @ pc, s=26, color=col, alpha=0.55 if col != INK else 0.45, lw=0, zorder=2)
            ax.scatter((Z @ off).mean(), 0.86, s=150, marker='v', color=col, edgecolor='white', lw=1.5, zorder=3)
        h = [plt.Line2D([], [], marker='o', ls='', ms=8, color=col, label=lab) for _, col, lab in sets]
        ax.legend(handles=h, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=len(sets), fontsize=12, handletextpad=0.1, columnspacing=1.2)
        ax.text(0.99, 0.985, 'percentages: top-1 accuracy · triangles: cloud centres', transform=ax.transAxes, ha='right', va='top', fontsize=10.5, color=MUTED) if ax is axes[1] else None
        ax.set_title(title, loc='left', fontsize=15, fontweight=700, pad=38)
        ax.set_xlabel("Direction of the new subject's offset"); ax.set_ylim(-0.85, 1.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    axes[0].set_ylabel('First principal component')
    save(fig, 'embedding_offset')


IMG_DIR = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set/test_images'   # sorted folder order = test index


def attention_matrices(subjects=(3, 6, 8), images=(4, 71, 153), top=None, keep=None, name=None, note=None):
    """Electrode-attention matrices of the frozen ATM (fold 1 model, seed 3300; subject 1 is held out, the others were seen in
    training), 80-repetition test trials, 4 heads averaged, shared token left out. Rows: attending electrode; columns: attended
    electrode. `top`: keep the electrodes receiving the most attention over all subjects and test images; `keep`: explicit indices."""
    from matplotlib.colors import PowerNorm
    from PIL import Image
    d = np.load(f'{HERE}/fig_data.npz'); A = d['A_own']
    if keep is None:
        keep = np.sort(np.argsort(A[:, :, 1:, 1:].astype(float).mean((0, 1, 2)))[::-1][:top]) if top else np.arange(63)
    keep = np.asarray(keep); n, full = len(keep), len(keep) == 63
    folders = sorted(os.listdir(IMG_DIR))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('v', ['#FFFFFF', '#E4DFF7', '#A497E3', VIOLET, '#2B1C73'])
    norm = PowerNorm(0.5, vmin=0, vmax=0.4)
    groups = [(0, 15, 'F'), (15, 26, 'FC'), (26, 35, 'C'), (35, 46, 'CP'), (46, 55, 'P'), (55, 63, 'PO/O')]
    R, C = len(subjects), len(images)
    cell = 3.5 if C <= 3 else 2.75
    fig = plt.figure(figsize=(cell * C + 2.1, cell * (R + 0.45) + 2.3))
    gs = fig.add_gridspec(R + 1, C + 1, height_ratios=[0.45] + [1] * R, width_ratios=[1] * C + [0.05], wspace=0.08, hspace=0.2 if full else 0.24,
                          left=0.12, right=0.93, top=0.97, bottom=0.07 if full else 0.11)
    for j, im in enumerate(images):
        ax = fig.add_subplot(gs[0, j])
        f = os.path.join(IMG_DIR, folders[im]); img = Image.open(os.path.join(f, sorted(os.listdir(f))[0])).convert('RGB')
        w, h = img.size; m = min(w, h); img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(folders[im].split('_', 1)[1].replace('_', ' '), fontsize=14, fontweight=600, pad=8)
    lab = lambda: [CH[c] for c in keep]
    for i, sub in enumerate(subjects):
        for j, im in enumerate(images):
            M = A[sub - 1, im].astype(float)[1:, 1:][np.ix_(keep, keep)]
            ax = fig.add_subplot(gs[i + 1, j])
            ax.imshow(M, cmap=cmap, norm=norm, interpolation='nearest', extent=(-0.5, n - 0.5, n - 0.5, -0.5))
            top_pad = 6 * n / 63 if n > 10 else 0.9
            ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5 - top_pad)
            k = int(M.mean(0).argmax())
            ax.plot([k, k], [-0.5, -0.5 - 0.3 * top_pad], color=INK, lw=1.2, clip_on=False)
            ax.text(k, -0.5 - 0.36 * top_pad, CH[keep[k]], ha='center' if k < n - 1 or n < 63 else 'right', va='bottom', fontsize=11, fontweight=600)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_bounds(-0.5, n - 0.5); ax.spines[sp].set_color(SOFT)
            if not full:
                ax.set_xticks(range(n), lab() if i == R - 1 else [], fontsize=10, rotation=90)
                ax.set_yticks(range(n), lab() if j == 0 else [], fontsize=10)
                ax.tick_params(length=0)
            else:
                ticks = [(a + b - 1) / 2 for a, b, _ in groups]; labels = [g for _, _, g in groups]
                ax.set_xticks(ticks, labels if i == R - 1 else [], fontsize=10.5)
                ax.set_yticks(ticks, labels if j == 0 else [], fontsize=10.5)
                ax.set_xticks([a - 0.5 for a, _, _ in groups[1:]], minor=True); ax.set_yticks([a - 0.5 for a, _, _ in groups[1:]], minor=True)
                ax.tick_params(which='major', length=0); ax.tick_params(which='minor', length=4, color=SOFT)
            if j == 0:
                ax.text(-0.5 - (22 * n / 63 if n > 10 else 0.42 * n), (n - 1) / 2, f'Subject {sub}', rotation=90, ha='center', va='center', fontsize=15, fontweight=700)
    cax = fig.add_subplot(gs[1:, C])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax)
    cb.set_ticks([0, 1 / 64, 0.1, 0.2, 0.4], labels=['0', 'uniform', '0.1', '0.2', '0.4']); cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=11); cb.set_label('Attention weight', color=MUTED, fontsize=12)
    lines = ['Attended electrode (columns)  ·  attending electrode (rows)  ·  electrodes ordered front to back']
    if note or top:
        lines.append(note or f'Rows and columns: the {n} electrodes that receive the most attention on average (all ten subjects, all 200 test images); weights from the full 63-electrode attention.')
    lines.append('Label above each matrix: the most-attended electrode. Frozen ATM, four heads averaged, 80-repetition test trials.')
    for r, line in enumerate(lines):
        fig.text(0.525, (0.15 + 0.27 * (len(lines) - 1 - r)) / fig.get_figheight(), line, ha='center', color=MUTED, fontsize=12 if r == 0 else 10.5)
    save(fig, name or (f'attention_matrices_top{top}' if top else 'attention_matrices'))


def attention_matrices_top16():
    attention_matrices(top=16)


def attention_matrices_po8():
    attention_matrices(subjects=(3, 7, 6, 8, 5), images=(4, 71, 153, 91, 150), keep=np.arange(55, 63), name='attention_matrices_po8',
                       note='Rows and columns: the eight parieto-occipital and occipital electrodes, which receive about half of all attention; weights from the full 63-electrode attention.')


def key_gates_head(unit=953, a=3, b=6):
    """How the key acts in the projection head h + W2 GELU(h) (TSConv, fold 1, seed 3300; figures/head_data.py)."""
    from scipy.stats import norm as N
    from matplotlib.patches import FancyArrowPatch
    d = np.load(f'{HERE}/head_data.npz', allow_pickle=True)
    H = d['tsconv_H']
    rows = pd.DataFrame(d['tsconv_rows'], columns=d['tsconv_cols']).astype(float).mean()
    gelu = lambda h: h * N.cdf(h); slope = lambda h: N.cdf(h) + h * N.pdf(h)
    HEAD_F, HEAD_S, IO_F, IO_S, STEM_F, STEM_S = '#F6E3B6', '#A8741A', '#EEF1F4', '#6B7785', '#C2E4DF', '#1F7A70'
    FW, FH = 17, 11
    fig = plt.figure(figsize=(FW, FH))
    def panel(x, y, letter, title):
        fig.text(x, y, letter, fontsize=17, fontweight=800, va='baseline')
        fig.text(x + 0.02, y, title, fontsize=13.5, fontweight=700, va='baseline')

    # ---- A: schematic
    rect = [0.03, 0.58, 0.955, 0.385]
    ax = fig.add_axes(rect); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
    asp = (30 / (rect[3] * FH)) / (100 / (rect[2] * FW))
    def box(x, y, w, h, fc, ec, title='', sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec=ec, lw=1.4, mutation_aspect=asp))
        if title:
            ax.text(x + w / 2, y + h / 2 + (1.3 if sub else 0), title, ha='center', va='center', fontsize=13, fontweight=600)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha='center', va='center', fontsize=10.5, color=MUTED)
    def arr(x0, x1, y, label=None):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=14, lw=1.5, color=INK))
        if label:
            ax.text((x0 + x1) / 2, y + 1.1, label, ha='center', va='bottom', fontsize=10.5, color=MUTED, family='DejaVu Sans Mono')
    y0, hh = 15, 8; ym = y0 + hh / 2
    box(0.5, y0, 12.5, hh, IO_F, IO_S)
    ax.text(6.75, y0 + hh - 2.0, 'EEG trial', ha='center', va='center', fontsize=13, fontweight=600)
    for k, (lab, col) in enumerate((('key', VIOLET), ('image signal', TEAL))):
        ax.add_patch(plt.Rectangle((2.6, y0 + 3.3 - 2.0 * k), 0.9, 1.2, fc=col, ec='none'))
        ax.text(4.0, y0 + 3.9 - 2.0 * k, lab, fontsize=10.5, va='center')
    arr(13, 15.5, ym)
    box(15.5, y0, 12, hh, STEM_F, STEM_S, 'Encoder stem', 'convolution / attention')
    arr(27.5, 30.5, ym)
    ax.add_patch(plt.Rectangle((29.6, 0.6), 69.6, 28.6, fc='#FBF4E3', ec='none', zorder=-1))
    ax.text(30.4, 27.6, 'PROJECTION HEAD', fontsize=10.5, fontweight=700, color=HEAD_S, va='center')
    box(30.5, y0, 8, hh, HEAD_F, HEAD_S, 'Linear')
    arr(38.5, 41.5, ym, 'h')
    xg = 41.5; box(xg, y0, 16, hh + 4, HEAD_F, HEAD_S)
    ax.text(xg + 8, y0 + hh + 2.4, 'GELU, unit by unit', ha='center', fontsize=12.5, fontweight=600)
    gx = np.linspace(-3, 2, 100); o, sc = (xg + 2.2, y0 + 1.6), (11.6 / 5, 6.2 / 2.2)
    ax.plot(o[0] + (gx + 3) * sc[0], o[1] + (gelu(gx) + 0.2) * sc[1], color=INK, lw=1.6)
    for hv, col, lab, dx, dy, ha in ((1.0, AMBER, 'on', 0.9, -1.6, 'left'), (-2.2, VIOLET, 'off', 0, 1.1, 'center')):
        px, py = o[0] + (hv + 3) * sc[0], o[1] + (gelu(hv) + 0.2) * sc[1]
        ax.scatter(px, py, s=70, color=col, edgecolor='white', lw=1.5, zorder=5)
        ax.text(px + dx, py + dy, lab, fontsize=10.5, fontweight=600, ha=ha)
    arr(57.5, 60.5, ym)
    box(60.5, y0, 9, hh, HEAD_F, HEAD_S, 'Linear', 'W₂')
    arr(69.5, 72.6, ym)
    ax.add_patch(plt.Circle((74, ym), 1.4 / asp ** 0.5, fc='white', ec=INK, lw=1.4)); ax.text(74, ym, '+', ha='center', va='center', fontsize=14)
    ax.plot([40, 40, 74, 74], [ym, y0 - 2.4, y0 - 2.4, ym - 2.4], color=INK, lw=1.4)
    ax.add_patch(FancyArrowPatch((74, ym - 2.6), (74, ym - 1.45), arrowstyle='-|>', mutation_scale=12, lw=0, color=INK))
    ax.text(57, y0 - 1.9, 'residual: h', ha='center', fontsize=10.5, color=MUTED, family='DejaVu Sans Mono')
    arr(75.4, 78, ym)
    box(78, y0, 10, hh, HEAD_F, HEAD_S, 'LayerNorm', '+ projector')
    arr(88, 90.5, ym)
    box(90.5, y0, 8.2, hh, IO_F, IO_S, 'EEG', 'embedding')
    ax.text(32, 8.6, 'h = μ + c', fontsize=15, fontweight=700)
    for k, (col, sym, txt) in enumerate(((VIOLET, 'μ', "the subject's mean input, set by the key: where each unit sits on the GELU"),
                                         (TEAL, 'c', 'the image signal, which a unit passes with gain  g = GELU′(μ)'))):
        yy = 5.0 - 2.9 * k
        ax.add_patch(plt.Rectangle((32, yy - 0.6), 0.55, 1.2, fc=col, ec='none'))
        ax.text(33.3, yy, sym, fontsize=12, fontweight=700, va='center'); ax.text(35.2, yy, txt, fontsize=11.5, va='center')
    ax.text(97.8, 7.4, r'image signal is read out by  $\mathbf{I + W_2\,diag(g)}$', ha='right', fontsize=14.5)
    ax.text(97.8, 3.9, 'a different readout matrix for every subject', ha='right', fontsize=12, color=MUTED)
    ax.text(0.5, 7.5, 'Up to the head, the key adds the same\nvector to the head input for every image\n(a wrong key: r = .98 with a constant shift).', fontsize=11, color=MUTED, va='center', linespacing=1.4)
    fig.text(0.012, 0.955, 'A', fontsize=17, fontweight=800)

    # ---- B: one unit, two subjects
    panel(0.012, 0.525, 'B', 'One unit of the head, 200 test images')
    ax = fig.add_axes([0.06, 0.22, 0.2, 0.27]); axb = fig.add_axes([0.06, 0.12, 0.2, 0.09], sharex=ax); axr = fig.add_axes([0.265, 0.22, 0.04, 0.27], sharey=ax)
    xs = np.linspace(-4.6, 3.2, 300)
    ax.plot(xs, gelu(xs), color=INK, lw=2, zorder=2)
    bins, obins = np.linspace(-4.6, 3.2, 40), np.linspace(-0.25, 3.2, 40)
    fmt = lambda v: '≈ 0' if abs(v) < 0.15 else f'≈ {v:.0f}'
    for k, (s_, col) in enumerate(((a, AMBER), (b, VIOLET))):
        h = H[s_ - 1, :, unit]
        ax.scatter(h, gelu(h), s=26, color=col, alpha=0.8, edgecolor='white', lw=0.5, zorder=3)
        axb.hist(h, bins=bins, color=col, alpha=0.8, lw=0); axr.hist(gelu(h), bins=obins, color=col, alpha=0.8, lw=0, orientation='horizontal')
        yy = 2.62 - 0.9 * k
        ax.scatter(-4.25, yy - 0.1, s=60, color=col, edgecolor='white', lw=1)
        ax.text(-4.0, yy, f'subject {s_}: slope {fmt(slope(h).mean())}\n' + ('image signal passes' if k == 0 else 'image signal blocked'), fontsize=11.5, fontweight=600, va='top')
    ax.set_ylim(-0.4, 2.9); ax.set_xlim(-4.6, 3.2); ax.set_ylabel('GELU output'); ax.tick_params(labelbottom=False)
    axb.set_xlabel('Unit input h'); axb.set_yticks([]); axb.spines['left'].set_visible(False); axb.invert_yaxis(); axb.spines['bottom'].set_visible(False); axb.spines['top'].set_visible(True); axb.xaxis.tick_bottom()
    axr.set_xticks([]); axr.spines['bottom'].set_visible(False); axr.tick_params(labelleft=False)

    # ---- C: gain pattern per subject
    panel(0.335, 0.525, 'C', 'Which units pass the image signal')
    ax = fig.add_axes([0.435, 0.17, 0.235, 0.29])
    G = slope(H).mean(1)
    sw = np.where(G.max(0) - G.min(0) > 0.6)[0]
    Gs = np.clip(G[:, sw], 0, 1)
    order = np.lexsort((-Gs.max(0), Gs.argmax(0)))
    ramp = mpl.colors.LinearSegmentedColormap.from_list('t', ['#FFFFFF', '#CDEDE6', '#5FC2AC', TEAL, '#0A5446'])
    ax.imshow(Gs[:, order], aspect='auto', cmap=ramp, vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(10), [f'Subject {s}' + (' (held out)' if s == 1 else '') for s in range(1, 11)], fontsize=11)
    ax.set_xticks([]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    k = int(np.where(sw[order] == unit)[0][0])
    ax.annotate('unit in B', (k, -0.55), xytext=(k + 3, -1.25), ha='left', va='center', fontsize=10.5, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1, mutation_scale=10, shrinkA=0), annotation_clip=False)
    ax.set_xlabel(f'The {len(sw)} head units (of {H.shape[-1]}) whose gain changes most between subjects', fontsize=11)
    cax = fig.add_axes([0.435, 0.075, 0.235, 0.014])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 1), ramp), cax=cax, orientation='horizontal')
    cb.set_ticks([0, 0.5, 1], labels=['0  blocked', '0.5', '1  passed']); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=10.5)
    fig.text(0.435 + 0.1175, 0.096, 'gain: mean GELU slope over the 200 test images', ha='center', fontsize=10.5, color=MUTED)

    # ---- D: what a wrong key breaks
    panel(0.705, 0.525, 'D', 'What a wrong key breaks')
    ax = fig.add_axes([0.84, 0.12, 0.145, 0.37])
    items = [('Own key', rows['own'], None), ("Another subject's key", rows['input_swap'], None), ('…applied only as a shift\nof the head input', rows['h_shift'], None),
             ('Head linearised at the\nsubject\'s operating point', rows['lin_own'], None), ("…with the other subject's\noffset", rows['wrong_offset_own_gain'], TEAL),
             ("…with the other subject's\ngains", rows['own_offset_wrong_gain'], VIOLET)]
    for y, (lab, v, col) in zip((6.2, 5.2, 4.2, 2.6, 1.6, 0.6), items):
        ax.barh(y, v, height=0.62, color=col or '#9AA5B1', lw=0)
        ax.text(v - 1.5, y, f'{v:.0f}', ha='right', va='center', color='white', fontsize=12.5, fontweight=700)
        ax.text(-2.5, y, lab, ha='right', va='center', fontsize=11, linespacing=1.1)
    ax.set_xlim(0, 72); ax.set_ylim(0, 7); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel('Top-1 (%), seen subjects')

    fig.text(0.012, 0.008, 'TSConv, validation-selected, fold 1 (subject 1 held out). D: the image signal of one seen subject decoded with the key of another, mean over 72 pairs; '
             'linearised head: GELU replaced by its tangent at the subject\'s mean input.\nSame split in ATM\'s head (74.8 linearised, 74.7 with the other offset, 67.2 with the other gains); '
             'ATM also uses the key in its attention and feed-forward layers.', fontsize=10.5, color=MUTED, va='bottom', linespacing=1.45)
    save(fig, 'key_gates_head')

_THUMBS = {}


def thumbs(px=56):
    """The 200 test images, centre-cropped, px x px, RGB float in [0, 1] (test index order)."""
    if px not in _THUMBS:
        from PIL import Image
        out = []
        for f in sorted(os.listdir(IMG_DIR)):
            img = Image.open(os.path.join(IMG_DIR, f, sorted(os.listdir(os.path.join(IMG_DIR, f)))[0])).convert('RGB')
            w, h = img.size; m = min(w, h)
            out.append(np.asarray(img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((px, px)), dtype=float) / 255)
        _THUMBS[px] = np.stack(out)
    return _THUMBS[px]


def hit_mosaic(hits, lost=None, rows=10, cols=20, px=56, gap=3):
    """One image: the 200 test images, correct retrievals in colour, misses faded to grey; `lost` gets a coloured frame."""
    T_ = thumbs(px)
    H_, W_ = rows * (px + gap) - gap, cols * (px + gap) - gap
    M = np.ones((H_, W_, 3))
    frame = np.array(mpl.colors.to_rgb(AMBER))
    for i in range(rows * cols):
        r, c = divmod(i, cols); y, x = r * (px + gap), c * (px + gap)
        t = T_[i]
        if not hits[i]:
            gray = t.mean(-1, keepdims=True).repeat(3, -1)
            t = 0.22 * gray + 0.78
        M[y:y + px, x:x + px] = t
        if lost is not None and lost[i]:
            for k in range(3):
                M[y + k, x:x + px] = frame; M[y + px - 1 - k, x:x + px] = frame; M[y:y + px, x + k] = frame; M[y:y + px, x + px - 1 - k] = frame
    return M


def person(ax, x, y, r, fc, ec=None, ls='-'):
    """A person glyph: head circle above a half-disc body, centred at (x, y), height about 3.6 r."""
    from matplotlib.patches import Circle, Wedge
    ax.add_patch(Circle((x, y + 0.75 * r), 0.62 * r, fc=fc, ec=ec or fc, lw=1.4, ls=ls))
    ax.add_patch(Wedge((x, y - 1.25 * r), 1.15 * r, 0, 180, fc=fc, ec=ec or fc, lw=1.4, ls=ls))


def key_swap_patching(seen_sub=3, held=1, donor=8, ch='Oz', img=0):
    """Trial − own mean ERP + another subject's mean ERP: top-1 change for a trained-on and the held-out subject. TSConv, fold 1; changes are means over all swaps."""
    from matplotlib.patches import FancyArrowPatch
    sd = np.load(f'{HERE}/stem_data.npz'); fd = np.load(f'{HERE}/fig_data.npz')
    seen = sd['seen']; mu, x = fd['mu'], fd['x']; c = CH.index(ch); t = np.arange(250) * 4
    hit = lambda s_, k: sd[f'hit_{s_}_{k}'].mean() * 100
    delta = {'seen': np.mean([hit(a, b) for a in seen for b in seen if a != b]) - np.mean([hit(a, a) for a in seen]),
             'held': np.mean([hit(held, b) for b in seen]) - hit(held, held)}
    fig = plt.figure(figsize=(17, 6.6))
    fig.text(0.5, 0.95, "An unseen subject's mean ERP is unused by the model", ha='center', fontsize=20, fontweight=700)
    cols = [(0.215, 'recorded trial'), (0.385, 'own mean ERP'), (0.555, "another subject's mean ERP")]
    for x0, lab in cols:
        fig.text(x0 + 0.055, 0.82, lab, ha='center', fontsize=13, fontweight=600)
    fig.text(0.89, 0.82, 'top-1 change', ha='center', fontsize=13, fontweight=600)
    for r_, (key, s_, label) in enumerate((('seen', seen_sub, 'trained on'), ('held', held, 'held out'))):
        y0 = 0.42 - 0.39 * r_
        ax = fig.add_axes([0.01, y0, 0.17, 0.36]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.3, 0.6), 5.9, 7.4, boxstyle='round,pad=0,rounding_size=0.5', fc='#F4F6F8', ec=SOFT, lw=1.2))
        ax.text(3.25, 8.4, 'training set', ha='center', fontsize=11.5, fontweight=600)
        for j in range(9):
            person(ax, 1.35 + 1.9 * (j % 3), 6.3 - 2.35 * (j // 3), 0.62, INK if (key == 'seen' and j == 4) else SOFT)
        if key == 'held':
            person(ax, 8.3, 4.0, 0.75, INK); ax.text(8.3, 1.9, label, ha='center', va='top', fontsize=14, fontweight=700)
        else:
            ax.text(7.5, 4.1, label, ha='left', va='center', fontsize=14, fontweight=700)
            ax.add_patch(FancyArrowPatch((7.2, 4.1), (4.3, 4.1), arrowstyle='-|>', mutation_scale=13, lw=1.4, color=INK))
        ym = y0 + 0.18
        for (x0, _), curve, col in zip(cols, (x[s_ - 1, img, c], mu[s_ - 1, c], mu[donor - 1, c]), (TEAL, VIOLET, AMBER)):
            a = fig.add_axes([x0, ym - 0.07, 0.11, 0.14]); a.plot(t, curve, color=col, lw=2.2); a.set_ylim(-1.2, 1.2); a.axis('off')
        for xs_, sym in ((0.355, '−'), (0.525, '+')):
            fig.text(xs_, ym, sym, ha='center', va='center', fontsize=26, fontweight=600)
        ax = fig.add_axes([0.675, ym - 0.08, 0.14, 0.16]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        ax.add_patch(FancyArrowPatch((0.2, 5), (2.2, 5), arrowstyle='-|>', mutation_scale=15, lw=1.6, color=INK))
        ax.add_patch(FancyBboxPatch((2.4, 2.6), 4.8, 4.8, boxstyle='round,pad=0,rounding_size=0.5', fc='#C2E4DF', ec='#1F7A70', lw=1.3))
        ax.text(4.8, 5, 'TSConv', ha='center', va='center', fontsize=13, fontweight=600)
        ax.add_patch(FancyArrowPatch((7.4, 5), (9.6, 5), arrowstyle='-|>', mutation_scale=15, lw=1.6, color=INK))
        d_ = round(delta[key], 1)
        fig.text(0.89, ym + 0.01, f'{d_:+.1f}'.replace('-', '−'), ha='center', va='center', fontsize=40, fontweight=700, color=VIOLET if key == 'seen' else MUTED)
        fig.text(0.89, ym - 0.085, 'points', ha='center', va='center', fontsize=13)
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.405, 0.405], color=GRID, lw=1))
    save(fig, 'key_swap_patching')

def stem_is_linear():
    """The TSConv stem: affine blocks and two ELUs used in their near-linear range (A); accuracy after a mean swap at each layer (C)."""
    from matplotlib.patches import FancyArrowPatch
    d = np.load(f'{HERE}/stem_data.npz'); sl = pd.read_csv(f'{HERE}/swap_layers.csv')
    fig = plt.figure(figsize=(17, 10.5))
    fig.text(0.02, 0.955, 'The stem: two affine blocks, two ELUs, one 1×1 convolution', fontsize=14.5, fontweight=700)
    # chain: chip, ELU plot, chip, ELU plot, chip
    yb, hb = 0.56, 0.33
    chips = [(0.02, 'Temporal conv\nAverage pool\nBatchNorm', 'affine'), (0.43, 'Spatial conv\nBatchNorm', 'affine'), (0.84, '1×1 conv', 'linear')]
    for x0, lab, kind in chips:
        ax = fig.add_axes([x0, yb + 0.08, 0.115, hb - 0.16]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle='round,pad=0,rounding_size=0.06', fc='#C2E4DF', ec='#1F7A70', lw=1.4))
        ax.text(0.5, 0.56, lab, ha='center', va='center', fontsize=12.5, fontweight=600, linespacing=1.3)
        ax.text(0.5, 0.13, kind, ha='center', va='center', fontsize=11.5)
    for xa, xb in ((0.137, 0.16), (0.398, 0.428), (0.548, 0.57), (0.808, 0.838)):
        fig.add_artist(FancyArrowPatch((xa, yb + hb / 2), (xb, yb + hb / 2), arrowstyle='-|>', mutation_scale=16, lw=1.6, color=INK, transform=fig.transFigure))
    for x0, key, lab in ((0.165, 'elu1_in', 'ELU 1'), (0.575, 'elu2_in', 'ELU 2')):
        frame = fig.add_axes([x0 - 0.005, yb - 0.005, 0.24, hb + 0.01]); frame.set_xticks([]); frame.set_yticks([])
        for sp in frame.spines.values():
            sp.set_visible(True); sp.set_color(AMBER); sp.set_linewidth(2)
        frame.patch.set_alpha(0)
        v = d[key]; y = np.where(v > 0, v, np.expm1(v)); A_ = np.polyfit(v, y, 1); r2 = 1 - np.var(y - np.polyval(A_, v)) / np.var(y)
        ax = fig.add_axes([x0 + 0.025, yb + 0.06, 0.2, hb - 0.12])
        xs = np.linspace(-2.5, 2.5, 200)
        ax.hist(np.clip(v, -2.5, 2.5), bins=70, range=(-2.5, 2.5), density=True, color=AMBER, alpha=0.5, lw=0)
        a2 = ax.twinx(); a2.plot(xs, np.where(xs > 0, xs, np.expm1(xs)), color=INK, lw=2.2); a2.plot(xs, np.polyval(A_, xs), color=INK, lw=1.3, ls=(0, (4, 3)))
        a2.set_ylim(-1.4, 2.6); a2.axis('off')
        ax.set_yticks([]); ax.set_xlim(-2.5, 2.5); ax.set_xticks([-2, -1, 0, 1, 2]); ax.tick_params(labelsize=10.5)
        for sp in ('left', 'right', 'top'):
            ax.spines[sp].set_visible(False)
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=13.5, fontweight=700, va='top')
        ax.text(0.99, 0.2, f'straight-line fit\nR² = {r2:.2f}', transform=ax.transAxes, fontsize=12, ha='right', va='bottom', linespacing=1.3)
        print(lab, 'R2 of a straight line on the real inputs', round(r2, 3))
    fig.text(0.165, yb - 0.04, 'amber: real inputs of the ELU   ·   solid: ELU   ·   dashed: straight-line fit', fontsize=11.5)
    # C: wrong mean ERP at the input, own mean activation restored at one layer (MI_LOG E07 C on this model)
    fig.text(0.02, 0.43, 'Wrong mean ERP at the input, own mean restored at one layer', fontsize=14.5, fontweight=700)
    names = ['input', 'after ELU 1', 'after ELU 2', 'stem output', 'head input h', 'after GELU block', 'after LayerNorm', 'embedding']
    ax = fig.add_axes([0.07, 0.08, 0.72, 0.3])
    xs = np.arange(len(names))
    ax.axvspan(-0.5, 3.5, color='#EAF5F3', lw=0, zorder=0); ax.axvspan(3.5, len(names) - 0.5, color='#FBF4E3', lw=0, zorder=0)
    ax.text(1.5, 0.03, 'stem', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=12, fontweight=700, color='#1F7A70')
    ax.text(5.5, 0.03, 'projection head', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=12, fontweight=700, color='#A8741A')
    for held_, col, lab in ((False, INK, 'trained-on subjects'), (True, VIOLET, 'held-out subject')):
        g_ = sl[sl.held == held_]; ys = g_[['restore ' + n for n in names]].mean().values; own, wrong = g_.own.mean(), g_.wrong.mean()
        ax.axhspan(wrong, own, color=col, alpha=0.07, lw=0, zorder=1)
        ax.plot(xs, ys, color=col, lw=2.2, zorder=3, solid_joinstyle='round'); ax.scatter(xs, ys, s=80, color=col, edgecolor='white', lw=2, zorder=4)
        ax.text(len(names) - 0.35, own, lab, va='bottom', fontsize=12.5, fontweight=600, color=INK)
        ax.text(len(names) - 0.35, own - 0.4, f'own mean ERP: {own:.1f} %\nwrong mean ERP: {wrong:.1f} %', va='top', fontsize=11.5, color=INK, linespacing=1.3)
        for i in (0, 4, 5):
            ax.text(xs[i] + 0.08, ys[i] + (1.0 if not held_ else -1.2), f'{round(ys[i], 1):.0f}', fontsize=12, fontweight=600, va='bottom' if not held_ else 'top')
    ax.set_xticks(xs, ['restored at\n' + n if i == 0 else n for i, n in enumerate(names)], fontsize=11.5, color=INK); ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(names) - 0.5); ax.set_ylim(30, 75); ax.set_ylabel('Top-1 (%)')
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, 'stem_is_linear')


def key_gates_head_bc(tags=(('tsconv', 953, 'key_gates_head_bc'), ('tsconv_bb128', None, 'key_gates_head_bc_bb128')), a=3, b=6):
    """One figure per model: one head unit for two subjects (left) and the gain of every head unit for every subject (right)."""
    from scipy.stats import norm as N
    d = np.load(f'{HERE}/head_data.npz', allow_pickle=True)
    gelu = lambda h: h * N.cdf(h); slope = lambda h: N.cdf(h) + h * N.pdf(h)
    ramp = mpl.colors.LinearSegmentedColormap.from_list('t', ['#FFFFFF', '#CDEDE6', '#5FC2AC', TEAL, '#0A5446'])
    fmt = lambda v: '≈ 0' if abs(v) < 0.15 else f'{v:.1f}'
    for tag, unit, name in tags:
        if f'{tag}_H' not in d.files:
            continue
        H = d[f'{tag}_H']; G = slope(H).mean(1); D = H.shape[-1]
        if unit is None:
            unit = int(np.argmax(np.abs(G[a - 1] - G[b - 1]) * np.minimum(H[a - 1].std(0), H[b - 1].std(0))))
        fig = plt.figure(figsize=(17, 5.6))
        # left: one unit, two subjects
        fig.text(0.012, 0.93, 'One unit of the head, 200 test images', fontsize=13.5, fontweight=700)
        ax = fig.add_axes([0.05, 0.38, 0.2, 0.47]); axb = fig.add_axes([0.05, 0.1, 0.2, 0.26], sharex=ax); axr = fig.add_axes([0.255, 0.38, 0.04, 0.47], sharey=ax)
        hs = [H[s_ - 1, :, unit] for s_ in (a, b)]
        lo, hi = min(h.min() for h in hs) - 0.3, max(h.max() for h in hs) + 0.3
        xs = np.linspace(lo, hi, 300); ax.plot(xs, gelu(xs), color=INK, lw=2, zorder=2)
        ytop = gelu(hi) * 1.12 + 0.3
        bins, obins = np.linspace(lo, hi, 40), np.linspace(-0.25, ytop, 40)
        for k, (s_, col, h) in enumerate(((a, AMBER, hs[0]), (b, VIOLET, hs[1]))):
            ax.scatter(h, gelu(h), s=24, color=col, alpha=0.8, edgecolor='white', lw=0.5, zorder=3)
            axb.hist(h, bins=bins, color=col, alpha=0.8, lw=0); axr.hist(gelu(h), bins=obins, color=col, alpha=0.8, lw=0, orientation='horizontal')
            g_ = slope(h).mean(); yy = ytop * (0.93 - 0.33 * k)
            ax.scatter(lo + 0.035 * (hi - lo), yy - 0.035 * ytop, s=55, color=col, edgecolor='white', lw=1)
            ax.text(lo + 0.07 * (hi - lo), yy, f'subject {s_}: slope {fmt(g_)}\n' + ('image signal passes' if g_ > 0.5 else 'image signal blocked' if g_ < 0.1 else 'image signal mostly blocked' if g_ < 0.35 else 'image signal partly passed'),
                    fontsize=11, fontweight=600, va='top')
        ax.set_xlim(lo, hi); ax.set_ylim(-0.35, ytop); ax.set_ylabel('GELU output'); ax.tick_params(labelbottom=False)
        axb.set_xlabel('Unit input h'); axb.set_yticks([]); axb.spines['left'].set_visible(False); axb.invert_yaxis(); axb.spines['bottom'].set_visible(False); axb.spines['top'].set_visible(True)
        axr.set_xticks([]); axr.spines['bottom'].set_visible(False); axr.tick_params(labelleft=False)
        # right: every unit
        fig.text(0.335, 0.93, f'Which units pass the image signal: all {D} head units', fontsize=13.5, fontweight=700)
        sw = (G.max(0) - G.min(0)) > 0.6; Gc = np.clip(G, 0, 1)
        o_sw = np.where(sw)[0][np.lexsort((-Gc[:, sw].max(0), Gc[:, sw].argmax(0)))]
        o_st = np.where(~sw)[0][np.argsort(-Gc[:, ~sw].mean(0))]
        ax = fig.add_axes([0.435, 0.24, 0.55, 0.58])
        ax.imshow(Gc[:, np.concatenate([o_sw, o_st])], aspect='auto', cmap=ramp, vmin=0, vmax=1, interpolation='nearest')
        ax.axvline(len(o_sw) - 0.5, color=INK, lw=1.2)
        ax.text((len(o_sw) - 1) / 2, -0.75, f'differ between subjects: {len(o_sw)}', ha='center', fontsize=11, fontweight=600)
        ax.text(len(o_sw) + (len(o_st) - 1) / 2, -0.75, f'same for every subject: {len(o_st)}', ha='center', fontsize=11, fontweight=600)
        ax.set_yticks(range(10), [f'Subject {s_}' + (' (held out)' if s_ == 1 else '') for s_ in range(1, 11)], fontsize=10.5)
        ax.set_xticks([]); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cax = fig.add_axes([0.435 + 0.175, 0.1, 0.2, 0.025])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 1), ramp), cax=cax, orientation='horizontal')
        cb.set_ticks([0, 0.5, 1], labels=['0  blocked', '0.5', '1  passed']); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=10.5)
        fig.text(0.435 + 0.165, 0.112, 'gain (mean GELU slope)', ha='right', va='center', fontsize=11, color=MUTED)
        save(fig, name)

def recentring_diagram():
    """Real re-centring vs in-trial simulated re-centring, one diagram. Waveforms: real 80-repetition trials at Oz (fold 1);
    embedding inset: real ATM embeddings of the held-out subject (fold 1). Accuracies: ATM with three heads (results_ladder)."""
    from matplotlib.patches import FancyArrowPatch, Circle
    fd = np.load(f'{HERE}/fig_data.npz'); x, mu, seen = fd['x'], fd['mu'], fd['seen']; c = CH.index('Oz')
    W, H_ = 100, 58
    fig = plt.figure(figsize=(18, 18 * H_ / W))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.axis('off')
    ATM_F, ATM_S, OP_F = '#ECE8FA', '#5946B0', '#FFFFFF'

    def inset(x0, y0, w, h):
        a = fig.add_axes([x0 / W, y0 / H_, w / W, h / H_]); a.axis('off'); return a

    def wave(x0, y0, w, h, curve, col, lw=1.8, alpha=1):
        a = inset(x0, y0, w, h); a.plot(curve, color=col, lw=lw, alpha=alpha); a.set_ylim(-1.25, 1.25); return a

    def chip(x0, y0, w, h, label, fc=ATM_F, ec=ATM_S, fs=12.5):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.7', fc=fc, ec=ec, lw=1.4))
        ax.text(x0 + w / 2, y0 + h / 2, label, ha='center', va='center', fontsize=fs, fontweight=600, linespacing=1.15)

    def arrow(p0, p1, col=INK, lw=1.5, style='-|>', rad=0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, lw=lw, color=col, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))

    def op(xc, yc, sym, r=1.5):
        ax.add_patch(Circle((xc, yc), r, fc=OP_F, ec=INK, lw=1.5)); ax.text(xc, yc - 0.1, sym, ha='center', va='center', fontsize=17, fontweight=600)

    def label(xc, yc, text, fs=11.5, ha='center', va='bottom', w=600):
        ax.text(xc, yc, text, ha=ha, va=va, fontsize=fs, fontweight=w, linespacing=1.2)

    q = x[0, 0, c]                                            # held-out subject 1, one test image
    K = mu[seen - 1].mean(0)[c]
    per = (q - K).reshape(5, 50).mean(0); per = per - per.mean()
    k_est = K + np.tile(per, 5)
    sigs = [x[s_ - 1, j, c] - mu[s_ - 1, c] for s_, j in ((2, 1), (4, 2), (7, 3))]
    yM, yT, yB = 30, 46.5, 12.5                               # rows: scoring, real, simulated

    # ---- middle row: scoring a query
    ax.text(1, yM + 6.6, 'Scoring a query', fontsize=14.5, fontweight=700)
    ax.text(24, yM + 6.6, '35.5 %', fontsize=24, fontweight=700, color=INK, va='baseline')
    ax.text(1, yM + 4.4, 'no re-centring', fontsize=12.5)
    wave(1, yM - 2.5, 10, 5, q, INK); label(6, yM - 4.6, 'query trial', va='top')
    arrow((11.5, yM), (14, yM)); chip(14, yM - 2.6, 6.5, 5.2, 'ATM')
    arrow((20.5, yM), (24.5, yM)); label(26.2, yM, 'e(x)', fs=13.5, va='center', w=700)
    arrow((28, yM), (57.5, yM)); op(59, yM, '−')
    arrow((60.5, yM), (64.5, yM)); op(66, yM, '+')
    label(66, yM - 2.2, 'anchor:\nmean image\nembedding', va='top', fs=11)
    arrow((67.5, yM), (70.5, yM))
    T_ = thumbs(96)
    for j, im in enumerate((4, 71, 153)):
        a = inset(71 + j * 4.6, yM - 2.1, 4.2, 4.2 * 18 / 18 * (W / H_) * (H_ / W)); a.imshow(T_[im])
        if j == 0:
            for sp in ('left', 'right', 'top', 'bottom'):
                a.spines[sp].set_visible(True); a.spines[sp].set_color(TEAL); a.spines[sp].set_linewidth(3)
            a.axis('on'); a.set_xticks([]); a.set_yticks([])
    label(77.6, yM + 2.6, 'nearest image', fs=11.5)

    # ---- top row: real re-centring
    ax.text(1, yT + 8.4, 'Real re-centring', fontsize=14.5, fontweight=700)
    ax.text(1, yT + 6.2, 'needs 20–50 unlabelled trials of the new subject', fontsize=12.5)
    ax.text(24, yT + 8.4, '52.6 %', fontsize=24, fontweight=700, color=VIOLET, va='baseline')
    for j in range(4):
        wave(1 + 0.8 * j, yT - 2.8 + 0.9 * j, 9, 4.5, x[0, j + 1, c], INK, lw=1.3, alpha=0.45 + 0.18 * j)
    label(6.5, yT - 4.2, 'unlabelled trials', va='top')
    arrow((14.2, yT), (16.5, yT)); chip(16.5, yT - 2.6, 6.5, 5.2, 'ATM')
    arrow((23, yT), (26, yT)); chip(26, yT - 2.6, 8, 5.2, 'average', fc=OP_F, ec=INK)
    ax.plot([34, 59], [yT, yT], color=INK, lw=1.5); arrow((59, yT), (59, yM + 1.6))
    label(58.2, (yT + yM) / 2 + 1, 'real\ncentre', ha='right', va='center', fs=12.5, w=700)

    # ---- bottom row: in-trial simulated re-centring
    ax.text(1, yB + 9.6, 'In-trial simulated re-centring', fontsize=14.5, fontweight=700)
    ax.text(1, yB + 7.4, 'needs only the query', fontsize=12.5)
    ax.text(24, yB + 9.6, '48.5 %', fontsize=24, fontweight=700, color=VIOLET, va='baseline')
    wave(1, yB - 2.5, 10, 5, q, INK); label(6, yB - 4.6, 'query trial', va='top')
    arrow((11.5, yB), (13.5, yB))
    ax.add_patch(FancyBboxPatch((13.5, yB - 3.4), 10.5, 6.8, boxstyle='round,pad=0,rounding_size=0.7', fc=OP_F, ec=INK, lw=1.4))
    a = inset(14.2, yB - 1.3, 9.1, 4.0)
    for p_ in range(5):
        a.plot((q - K)[50 * p_:50 * p_ + 50], color=INK, lw=0.9, alpha=0.35)
    a.plot(per, color=VIOLET, lw=2.2); a.set_ylim(-1.1, 1.1)
    label(18.75, yB - 2.9, 'average its five\n200 ms periods', va='bottom', fs=10.5)
    arrow((24, yB), (26, yB))
    wave(26, yB - 2.2, 8, 4.4, k_est, VIOLET); label(30, yB + 2.6, 'its mean ERP,\nestimated', fs=11)
    arrow((34.3, yB), (36.2, yB)); op(37.7, yB, '+')
    for j, sg in enumerate(sigs):
        wave(33.2 + 0.7 * j, 1.0 + 0.7 * j, 7.5, 3.4, sg * 1.6, TEAL, lw=1.2, alpha=0.5 + 0.25 * j)
    label(42.6, 1.5, 'image signals of\ntraining subjects (×16)', ha='left', va='bottom', fs=11)
    arrow((37.7, 5.6), (37.7, yB - 1.5))
    arrow((39.2, yB), (41, yB))
    for j, sg in enumerate(sigs):
        wave(41 + 0.7 * j, yB - 2.6 + 0.8 * j, 7.5, 4.2, k_est + sg, INK, lw=1.2, alpha=0.5 + 0.25 * j)
    label(45.5, yB + 4.0, '16 simulated\ntrials', fs=11)
    arrow((50, yB), (51.5, yB)); chip(51.5, yB - 2.6, 6.2, 5.2, 'ATM')
    chip(55.0, yB + 4.6, 8, 4.0, 'average', fc=OP_F, ec=INK)
    ax.plot([57.7, 59.0, 59.0], [yB, yB, yB + 4.6], color=INK, lw=1.5)
    arrow((59.0, yB + 8.6), (59.0, yM - 1.6))
    label(58.2, (yB + 8.6 + yM - 1.6) / 2, 'simulated\ncentre', ha='right', va='center', fs=12.5, w=700)

    # ---- embedding inset (real ATM embeddings, held-out subject)
    E, g, anc = fd['E_own'][0], fd['g'], fd['anchor']
    off = E.mean(0) - g.mean(0); off /= np.linalg.norm(off)
    Xc = np.concatenate([g - g.mean(0), E - E.mean(0)]); Xc -= np.outer(Xc @ off, off)
    pc = np.linalg.svd(Xc, full_matrices=False)[2][0]
    a = fig.add_axes([86.5 / W, 15 / H_, 12.5 / W, 30 / H_])
    P = lambda Z: (Z @ off, Z @ pc)
    a.scatter(*P(g), s=9, color=SOFT, lw=0, zorder=1)
    a.scatter(*P(E), s=9, color=VIOLET, alpha=0.55, lw=0, zorder=2)
    a.scatter(*P(E - E.mean(0) + anc), s=9, color=TEAL, alpha=0.55, lw=0, zorder=2)
    (cx, cy), (ax_, ay_) = P(E.mean(0)[None]), P((anc)[None])
    a.annotate('', (ax_[0], ay_[0]), (cx[0], cy[0]), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2, mutation_scale=16), zorder=5)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(SOFT)
    a.set_title('In the embedding space', fontsize=12.5, fontweight=700, pad=8)
    for k_, (col, lab) in enumerate(((SOFT, 'images'), (VIOLET, 'new subject'), (TEAL, 'new subject, re-centred'))):
        fig.text(86.5 / W, (11.8 - 2.0 * k_) / H_, '●', color=col, fontsize=13, va='center')
        fig.text(87.8 / W, (11.8 - 2.0 * k_) / H_, lab, fontsize=11, va='center')
    save(fig, 'recentring_diagram')


FIGS = {f.__name__: f for f in (key_swap_architectures, key_use_by_stage, results_ladder, attention_follows_mean_erp, attention_by_subject, key_steady_state, embedding_offset, attention_matrices, attention_matrices_top16, attention_matrices_po8, key_gates_head, key_swap_patching, stem_is_linear, key_gates_head_bc, recentring_diagram)}
if __name__ == '__main__':
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
