"""Fig. 1 schematic: the gallery as a fixed point cloud of THINGS-EEG-2 test images (a C shape), the unlabeled EEG
queries of a new subject as a rotated, squashed and noisy copy of it inside the gallery, and the same clouds after
registration. Synthetic positions and EEG traces, illustration only (no real embeddings).
One axes in data units for both panels, so they share a scale and the figure is cropped to its content.
Writes CVPR_paper/figures/clouds.pdf (+ .png preview).
usage: .venv/bin/python CVPR_paper/scripts/make_clouds_figure.py
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

IMG = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set/test_images'
OUT = 'CVPR_paper/figures/clouds'
INK, BLUE, RED = '#1d2340', '#2a78d6', '#d63a3a'
NAMES = ['banana', 'basketball', 'elephant', 'flamingo', 'strawberry', 'cat']
COLS = ['#e0a800', '#eb6834', '#2a78d6', '#d6336c', '#1baf7a', '#9b59b6']
TS = 0.48  # thumbnail side, data units
EW, EH = 0.37, 0.25  # EEG icon, data units
RG, RE = 1.2, 0.66  # radius of the gallery C and of the EEG C
W = 3.3  # inches (\columnwidth)


def thumb(name, size=180):
    im = Image.open(glob.glob(f'{IMG}/*_{name}/*.jpg')[0]).convert('RGB')
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS))


def rot(a):
    a = np.deg2rad(a)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def tube(ax, pts, **kw):
    """Thick smoothed line through the ordered points: the shape of the cloud (Chaikin corner cutting)."""
    for _ in range(4):
        q = np.concatenate([[0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b] for a, b in zip(pts[:-1], pts[1:])])
        pts = np.concatenate([pts[:1], q, pts[-1:]])
    ax.plot(*pts.T, solid_capstyle='round', solid_joinstyle='round', **kw)


def eeg(ax, x, y, col, seed):
    """EEG icon: a small recording tile with three coloured traces."""
    ax.add_patch(FancyBboxPatch((x - EW / 2, y - EH / 2), EW, EH, boxstyle='round,pad=0,rounding_size=0.04',
                                fc='white', ec=col, lw=0.9, zorder=6))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 50)
    for off in (0.27, 0.0, -0.27):
        s = np.convolve(rng.normal(size=58), np.ones(7) / 7, 'valid')[:50]
        s = s / (np.abs(s).max() + 1e-9)
        ax.plot(x - 0.4 * EW + 0.8 * EW * t, y + EH * (off + 0.13 * s), color=col, lw=0.55, zorder=7)


ang = np.deg2rad(np.linspace(60, 300, len(NAMES)))
U = np.c_[np.cos(ang), np.sin(ang)]
G = RG * U  # gallery, centred on the origin; it never moves
rng = np.random.default_rng(4)
E0 = (RE * U @ np.diag([1.0, 0.92])) @ rot(150).T + [0.03, -0.02] + rng.normal(0, 0.03, U.shape)  # before
E1 = RE * U + rng.normal(0, 0.02, U.shape)  # after: next to the image of its stimulus

# panel box around the gallery (data units); the second panel is the same box shifted by DX
PAD, TITLE, GAPX = 0.16, 0.36, 0.62
bx = (G[:, 0].min() - TS / 2 - PAD, G[:, 0].max() + TS / 2 + PAD)
by = (G[:, 1].min() - TS / 2 - PAD, G[:, 1].max() + TS / 2 + PAD + TITLE)
DX = bx[1] - bx[0] + GAPX
XL, YL = (bx[0] - 0.02, bx[1] + DX + 0.02), (by[0] - 0.02, by[1] + 0.02)
fig = plt.figure(figsize=(W, W * (YL[1] - YL[0]) / (XL[1] - XL[0])))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=XL, ylim=YL, aspect='equal')
ax.axis('off')

for k, (title, E) in enumerate((('Before calibration', E0), ('After GALA', E1))):
    dx = k * DX
    ax.add_patch(FancyBboxPatch((bx[0] + dx, by[0]), bx[1] - bx[0], by[1] - by[0], boxstyle='round,pad=0,rounding_size=0.15',
                                fc='white', ec='#c9ced8', lw=0.8, zorder=0))
    ax.text(bx[0] + dx + 0.14, by[1] - 0.2, title, fontsize=7.5, fontweight='bold', color=INK, va='center')
    tube(ax, G + [dx, 0], lw=17, color='#e4e7ee', zorder=1)
    tube(ax, E + [dx, 0], lw=13, color=BLUE, alpha=0.2, zorder=2)
    for (x, y), name, c in zip(G + [dx, 0], NAMES, COLS):
        ax.imshow(thumb(name), extent=[x - TS / 2, x + TS / 2, y - TS / 2, y + TS / 2], zorder=3)
        ax.add_patch(Rectangle((x - TS / 2, y - TS / 2), TS, TS, fill=False, ec=c, lw=1.0, zorder=4))
    for i, ((x, y), c) in enumerate(zip(E + [dx, 0], COLS)):
        eeg(ax, x, y, c, seed=20 + i)

# before: nearest-neighbour retrieval picks the wrong image (one link per retrieved image, at most two)
nn = np.linalg.norm(E0[:, None] - G[None], axis=2).argmin(1)
shown = {}
for i in np.flatnonzero(nn != np.arange(len(G))):
    shown.setdefault(nn[i], i)
for j, i in list(shown.items())[:2]:
    ax.plot(*np.c_[E0[i], G[j]], color=RED, lw=0.9, ls=(0, (2, 1.2)), zorder=5)
# before: the rotation that GALA estimates, in the middle of the EEG cloud
c0 = E0.mean(0) * 0.5
r, th = 0.2, np.deg2rad(np.linspace(110, 375, 60))  # counter-clockwise arc, head drawn on its end tangent
ax.plot(*(c0 + r * np.c_[np.cos(th), np.sin(th)]).T, color=BLUE, lw=1.3, solid_capstyle='butt', zorder=7)
end, tan = c0 + r * np.array([np.cos(th[-1]), np.sin(th[-1])]), np.array([-np.sin(th[-1]), np.cos(th[-1])])
nrm = np.array([tan[1], -tan[0]])
ax.add_patch(Polygon([end + 0.12 * tan, end + 0.06 * nrm, end - 0.06 * nrm], closed=True, fc=BLUE, ec='none', zorder=7))
# between the panels
ym = (by[0] + by[1]) / 2
ax.add_patch(FancyArrowPatch((bx[1] + 0.08, ym), (bx[0] + DX - 0.08, ym), arrowstyle='-|>', mutation_scale=15, lw=2.6,
                             color=INK))

fig.savefig(f'{OUT}.pdf', dpi=300)
fig.savefig(f'{OUT}.png', dpi=300)
print(f'wrote {OUT}.pdf ({W:.2f} x {fig.get_figheight():.2f} in), nearest image correct before: {(nn == np.arange(len(G))).sum()}/{len(G)}')
