"""Paper figure 1 (single column): the two standard evaluations (zero-shot, batch) on top, and the incremental
transductive setting below, drawn with THINGS-EEG-2 test images as the gallery.
EEG queries are drawn as waveform tiles (synthetic traces, illustration only). Everything is laid out in inches on one
axes, so the proportions are those of the printed figure. Writes CVPR_paper/figures/protocol.pdf.
usage: make_protocol_figure.py
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

IMG = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set/test_images'
OUT = 'CVPR_paper/figures'
BLUE, LBLUE, GREY, INK, SUB, GREEN = '#2a78d6', '#e6effa', '#9a9a9a', '#1d2340', '#4a4f5c', '#1baf7a'
QCOL = ['#2a78d6', '#eb6834', '#1baf7a', '#9b59b6']  # one colour per queried image, used on both sides
W, H = 3.3, 3.72
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis('off')


def thumb(name, size=160):
    im = Image.open(glob.glob(f'{IMG}/*_{name}/*.jpg')[0]).convert('RGB')
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS))


GAL = [thumb(n) for n in ('banana', 'antelope', 'basketball', 'bassoon', 'beaver', 'aircraft_carrier')]


def panel(x0, y0, w, h, title, sub, ec='#c9ced8', lw=0.8, fc='white'):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.07', fc=fc, ec=ec, lw=lw, zorder=1))
    ax.text(x0 + 0.10, y0 + h - 0.17, title, fontsize=9, fontweight='bold', color=INK, va='center', zorder=5)
    ax.text(x0 + 0.10, y0 + h - 0.325, sub, fontsize=6.3, color=SUB, va='top', linespacing=1.2, zorder=5)


def image(k, x, y, s, edge=None, alpha=1.0):
    ax.imshow(GAL[k], extent=[x - s / 2, x + s / 2, y - s / 2, y + s / 2], alpha=alpha, zorder=3, aspect='auto')
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, ec=edge or '#d5d8de', lw=1.3 if edge else 0.5, zorder=4))


def eeg(x, y, w, h, col=BLUE, seed=0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.03', fc=LBLUE, ec=col, lw=1.0, zorder=3))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 60)
    for off in (0.28, 0.0, -0.28):
        sig = np.convolve(rng.normal(size=70), np.ones(8) / 8, 'valid')[:60]
        ax.plot(x - 0.4 * w + 0.8 * w * t, y + h * (off + 0.11 * sig / (np.abs(sig).max() + 1e-9)), color=col, lw=0.55, zorder=4)


def link(x0, y0, x1, y1, col, lw=1.0, ls='-'):
    ax.plot([x0, x1], [y0, y1], color=col, lw=lw, ls=ls, zorder=2, solid_capstyle='round')


M, GAP = 0.02, 0.05                # outer margin, gap between panels
BH = 1.90                          # height of the incremental panel
TY, TH = M + BH + GAP, H - 2 * M - BH - GAP  # top row
PW = (W - 2 * M - GAP) / 2
S, TW = 0.215, 0.29                # gallery thumbnail and EEG tile in the top panels
ys = [TY + TH - d for d in (0.71, 0.965, 1.22, 1.475)]
# Zero-shot: one query scored alone against the gallery
x0 = M
panel(x0, TY, PW, TH, 'Zero-shot', 'Each query is decoded\nalone, without calibration')
qy = (ys[1] + ys[2]) / 2
eeg(x0 + 0.42, qy, TW, S, seed=1)
for k, y in enumerate(ys):
    image(k, x0 + PW - 0.36, y, S, edge=GREEN if k == 0 else None)
    link(x0 + 0.42 + TW / 2 + 0.02, qy, x0 + PW - 0.36 - S / 2 - 0.02, y, GREEN if k == 0 else GREY, lw=1.5 if k == 0 else 0.6,
         ls='-' if k == 0 else (0, (2, 2)))
# Batch: N queries for N images, one query per image, calibrated together
x0 = M + PW + GAP
panel(x0, TY, PW, TH, 'Batch', 'All queries at once, one\nquery per gallery image')
perm = [1, 0, 3, 2]
for i, y in enumerate(ys):
    eeg(x0 + 0.42, y, TW, S, col=QCOL[perm[i]], seed=10 + i)
    image(i, x0 + PW - 0.36, y, S, edge=QCOL[i])
    link(x0 + 0.42 + TW / 2 + 0.02, y, x0 + PW - 0.36 - S / 2 - 0.02, ys[perm[i]], QCOL[perm[i]], lw=1.2)
ax.add_patch(FancyBboxPatch((x0 + 0.42 - TW / 2 - 0.05, ys[-1] - S / 2 - 0.05), TW + 0.10, ys[0] - ys[-1] + S + 0.10,
                            boxstyle='round,pad=0,rounding_size=0.05', fill=False, ec=BLUE, lw=0.7, ls=(0, (3, 2)), zorder=2))
# Incremental: the gallery is known; queries arrive one at a time with unequal frequencies, some images are never
# queried; at every step the calibration is fitted on all the queries received so far
panel(M, M, W - 2 * M, BH, 'Incremental transductive calibration', '', ec=BLUE, lw=1.3, fc='#f8fbff')
seq = [0, 1, 0, 2, 0, 1]  # image queried at each step: 3 x, 2 x, 1 x; three gallery images never
GS, gy, cy = 0.40, M + 1.20, M + 0.905  # gallery thumbnail, its row, the row of query counts
gx = [W / 2 + 0.50 * (j - 2.5) for j in range(6)]
ax.text(W / 2, M + 1.50, 'Known gallery', fontsize=8.5, color=INK, ha='center', va='center')
for j, x in enumerate(gx):
    image(j, x, gy, GS, edge=QCOL[j] if j < 3 else '#b5b5b5', alpha=1.0 if j < 3 else 0.5)
    if j < 3:
        ax.text(x, cy, f'{seq.count(j)}$\\times$', fontsize=8.5, color=QCOL[j], ha='center', va='center', fontweight='bold')
ax.text(gx[4], cy, 'never', fontsize=8.5, color=GREY, ha='center', va='center')
for xa, xb in ((gx[3] - GS / 2, gx[4] - 0.23), (gx[4] + 0.23, gx[5] + GS / 2)):  # one rule under the three images never queried
    ax.plot([xa, xb], [cy] * 2, color='#b5b5b5', lw=0.7)
QW, QH, pitch, qy = 0.30, 0.22, 0.34, M + 0.685
xs = [M + 0.10 + pitch * (t + 0.5) for t in range(len(seq))]
for t, (x, m) in enumerate(zip(xs, seq)):
    eeg(x, qy, QW, QH, col=QCOL[m], seed=30 + t)
ax.text(xs[-1] + pitch * 0.8, qy, '...', fontsize=9, color=GREY, ha='center', va='center')
ax.add_patch(FancyArrowPatch((M + 0.09, M + 0.53), (W - M - 0.09, M + 0.53), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=INK, zorder=3))
ax.text(W - M - 0.11, M + 0.575, 'time', ha='right', fontsize=7.5, color=INK)
for i, (n, y) in enumerate(zip((2, 4, 6), (0.40, 0.275, 0.15))):  # the buffer of each refit: every query received so far
    x1 = xs[n - 1] + pitch / 2
    ax.add_patch(FancyBboxPatch((xs[0] - pitch / 2, M + y - 0.05), x1 - (xs[0] - pitch / 2), 0.10, boxstyle='round,pad=0,rounding_size=0.025',
                                fc=to_rgba(BLUE, 0.25 + 0.15 * i), ec='black', lw=0.6, zorder=2))
    ax.text(x1 + 0.05, M + y, f'calibration at step {n}', fontsize=7, color=BLUE, va='center')
fig.savefig(f'{OUT}/protocol.pdf')
fig.savefig(f'{OUT}/protocol.png', dpi=300)
