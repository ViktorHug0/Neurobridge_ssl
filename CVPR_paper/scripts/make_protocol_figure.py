"""Paper figure: the four evaluation regimes R0-R3, drawn with THINGS-EEG-2 test images as the gallery.
EEG queries are drawn as waveform tiles (synthetic traces, illustration only). Writes CVPR_paper/figures/protocol.pdf.
usage: make_protocol_figure.py
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

IMG = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set/test_images'
OUT = 'CVPR_paper/figures'
BLUE, LBLUE, GREY, LGREY, INK, GREEN = '#2a78d6', '#e6effa', '#9a9a9a', '#f3f3f3', '#1d2340', '#1baf7a'
QCOL = ['#2a78d6', '#eb6834', '#1baf7a', '#9b59b6']  # one colour per queried image, used on both sides
plt.rcParams.update({'font.size': 8})
W, H = 6.9, 2.15 * (1 - 0.09)  # bottom strip (y < 0.09) cropped
fig = plt.figure(figsize=(W, H))
PW = 0.245  # panel width in figure fraction


def thumb(name, size=160):
    im = Image.open(glob.glob(f'{IMG}/*_{name}/*.jpg')[0]).convert('RGB')
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS))


GAL = [thumb(n) for n in ('banana', 'antelope', 'basketball', 'bassoon', 'beaver', 'aircraft_carrier', 'basil')]


PANELS = []


class Panel:
    def __init__(self, i, title, sub):
        x0 = 0.008 + i * (PW + 0.003)
        self.ax = fig.add_axes([x0, 0.0, PW, 1.0])
        self.ax.set(xlim=(0, 1), ylim=(0.09, 1))
        self.ax.axis('off')
        self.r = (H / 0.91) / (W * PW)  # data-units ratio: 1 unit of y = r units of x, in inches
        self.ax.add_patch(FancyBboxPatch((0.02, 0.105), 0.96, 0.875, boxstyle='round,pad=0,rounding_size=0.04',
                                         fc='white', ec='#c9ced8', lw=0.8))
        PANELS.append(self)
        self.ax.text(0.07, 0.905, title, fontsize=9.5, fontweight='bold', color=INK, va='center')
        self.sub = self.ax.text(0.07, 0.815, sub, fontsize=7, color='#4a4f5c', va='center', linespacing=1.15)

    def image(self, k, x, y, h, alpha=1.0, edge=None):
        w = h * self.r
        self.ax.imshow(GAL[k], extent=[x - w / 2, x + w / 2, y - h / 2, y + h / 2], alpha=alpha, zorder=3, aspect='auto')
        if edge:
            self.ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=edge, lw=1.4, zorder=4))
        return x - w / 2, x + w / 2

    def eeg(self, x, y, h, col=BLUE, seed=0):
        w = h * self.r * 1.25
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.015',
                                         fc=LBLUE, ec=col, lw=1.1, zorder=3))
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, 60)
        for j, off in enumerate((0.28, 0.0, -0.28)):
            sig = np.convolve(rng.normal(size=70), np.ones(8) / 8, 'valid')[:60]
            sig = sig / (np.abs(sig).max() + 1e-9)
            self.ax.plot(x - w * 0.4 + 0.8 * w * t, y + h * (off + 0.11 * sig), color=col, lw=0.6, zorder=4)
        return x - w / 2, x + w / 2

    def link(self, x0, y0, x1, y1, col, lw=1.0, ls='-', alpha=1.0):
        self.ax.plot([x0, x1], [y0, y1], color=col, lw=lw, ls=ls, alpha=alpha, zorder=2, solid_capstyle='round')


ys4 = [0.655, 0.505, 0.355, 0.205]
hq = 0.115
# R0: one query scored alone against the gallery
p = Panel(0, 'R0  Zero-shot', 'Each query is decoded alone,\nwithout calibration')
_, qr = p.eeg(0.25, 0.43, hq, seed=1)
for k, y in enumerate(ys4):
    l, _ = p.image(k, 0.76, y, hq, edge=GREEN if k == 0 else None)
    p.link(qr + 0.01, 0.43, l - 0.01, y, GREEN if k == 0 else GREY, lw=1.6 if k == 0 else 0.7,
           ls='-' if k == 0 else (0, (2, 2)))
# R1: N queries vs the N images, one query per image, calibrated together
p = Panel(1, 'R1  Batch', 'All queries at once, one query\nper gallery image')
perm = [1, 0, 3, 2]
for i, y in enumerate(ys4):
    _, qr = p.eeg(0.25, y, hq, col=QCOL[perm[i]], seed=10 + i)
    l, _ = p.image(i, 0.76, y, hq, edge=QCOL[i])
    p.link(qr + 0.01, y, l - 0.01, ys4[perm[i]], QCOL[perm[i]], lw=1.3)
p.ax.add_patch(FancyBboxPatch((0.115, 0.13), 0.27, 0.6, boxstyle='round,pad=0,rounding_size=0.03', fill=False,
                              ec=BLUE, lw=0.8, ls=(0, (3, 2))))
# R2: queries arrive one by one from a menu of K images; repeats allowed; refits on the growing buffer
p = Panel(2, 'R2  Streaming', 'Queries arrive one by one\nfrom a menu of $K$ images')
menu = [0, 2, 4]
for j, k in enumerate(menu):
    p.image(k, 0.40 + j * 0.18, 0.62, 0.115, edge=QCOL[j])
p.ax.text(0.09, 0.62, 'menu', fontsize=6.5, color='#4a4f5c', va='center', style='italic')
seq = [0, 2, 1, 0, 2, 0]  # index into menu: repeats allowed, random order
xs = np.linspace(0.15, 0.85, len(seq))
for t, (x, m) in enumerate(zip(xs, seq)):
    p.eeg(x, 0.35, 0.075, col=QCOL[m], seed=30 + t)
ax = p.ax
ax.add_patch(FancyArrowPatch((0.08, 0.25), (0.94, 0.25), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=INK))
ax.text(0.92, 0.205, 'time', ha='right', fontsize=6.5, color=INK)
for x in (xs[1] + 0.07, xs[3] + 0.07, xs[5] + 0.07):
    ax.plot([x, x], [0.27, 0.43], color=BLUE, lw=0.9, ls=(0, (2, 1.5)))
ax.text(xs[1] + 0.07, 0.455, 'refit', ha='center', fontsize=6, color=BLUE)
ax.add_patch(Rectangle((0.075, 0.28), xs[3] + 0.07 - 0.075, 0.14, fc=BLUE, alpha=0.08, ec='none', zorder=1))
# R3: open gallery, D distractors that are never queried
p = Panel(3, 'R3  Open gallery', 'The gallery contains $D$ images\nthat are never queried')
for i, y in enumerate(ys4):
    _, qr = p.eeg(0.2, y, hq, col=QCOL[perm[i]], seed=50 + i)
    l, _ = p.image(i, 0.58, y, hq, edge=QCOL[i])
    p.link(qr + 0.01, y, l - 0.01, ys4[perm[i]], QCOL[perm[i]], lw=1.3)
for y, k in zip(ys4[:3], (4, 5, 6)):
    p.image(k, 0.84, y, hq, alpha=0.7, edge='#b5b5b5')
p.ax.text(0.84, 0.19, '$+D$', ha='center', fontsize=8, color=GREY, fontweight='bold')
p.ax.add_patch(FancyBboxPatch((0.745, 0.14), 0.19, 0.59, boxstyle='round,pad=0,rounding_size=0.03', fill=False,
                              ec=GREY, lw=0.8, ls=(0, (3, 2))))
fig.canvas.draw()  # every subtitle must stay inside its panel border (0.98 in panel coordinates)
for q in PANELS:
    bb = q.sub.get_window_extent().transformed(q.ax.transData.inverted())
    assert bb.x1 < 0.96, (q.sub.get_text(), bb.x1)
fig.savefig(f'{OUT}/protocol.pdf')
fig.savefig(f'{OUT}/protocol.png', dpi=220)
