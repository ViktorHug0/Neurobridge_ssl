"""Fig. 1: the GALA schema (figures/GALA.png, drawn by the authors) with a results panel appended on its right.
Bars: published inductive cross-subject top-1 (Table 1), SCORE and GALA stacked as zero-shot encoder + gain of the
label-free transductive calibration. SCORE: its own numbers (recovery-aware SAMGA, zero-shot 29.3, calibrated 53.2).
GALA: ATM, three seeds (Table 1). Update OURS when the ValCon numbers land.
Writes CVPR_paper/figures/teaser.png.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image

F = 'CVPR_paper/figures'
OURS = (32.2, 81.8)  # zero-shot, GALA (Table 1, ValCon)
BARS = [('UBP', 12.4, None), ('Neural-\nMCRL', 14.0, None), ('Neuro-\nBridge', 19.0, None),
        ('Shallow\nAlign.', 21.8, None), ('SCORE', 29.3, 53.2), ('GALA\n(ours)', *OURS)]
GREY, ZS, CAL, EDGE = '#9a9a9a', '#d6f3f7', '#f6d9c6', '#1d2340'

schema = Image.open(f'{F}/GALA.png').convert('RGBA')
H = schema.height
plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold'})
fig = plt.figure(figsize=(1150 / 200, H / 200), dpi=200)
fig.patches.append(FancyBboxPatch((0.02, 0.012), 0.96, 0.976, boxstyle='round,pad=0,rounding_size=0.03',
                                  transform=fig.transFigure, facecolor='#e2f4dc', edgecolor=EDGE, lw=2.2, zorder=-10))
fig.text(0.07, 0.915, 'Results', fontsize=30, fontweight='bold')
ax = fig.add_axes([0.21, 0.15, 0.74, 0.68])
ax.set_facecolor('white')
for i, (name, zs, cal) in enumerate(BARS):
    if cal is None:
        ax.bar(i, zs, color=GREY, edgecolor='black', lw=0.8, width=0.7)
        top = zs
    else:
        ax.bar(i, zs, color=ZS, edgecolor='black', lw=0.8, width=0.7)
        ax.bar(i, cal - zs, bottom=zs, color=CAL, edgecolor='black', lw=0.8, width=0.7)
        top = cal
    ax.text(i, top + 1.5, f'{top:.1f}', ha='center', va='bottom', fontsize=15)
ax.set_xticks(range(len(BARS)), [b[0] for b in BARS], fontsize=12.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Top-1 accuracy (%)', fontsize=19)
ax.tick_params(axis='y', labelsize=15)
ax.set_title('Unseen subjects, THINGS-EEG-2', fontsize=17, pad=10)
ax.grid(axis='y', lw=0.5, color='#e2e2e2'); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(facecolor=GREY, edgecolor='black', label='Published (no calibration)'),
                   Patch(facecolor=ZS, edgecolor='black', label='Zero-shot encoder'),
                   Patch(facecolor=CAL, edgecolor='black', label='Label-free calibration')],
          loc='upper left', fontsize=15, frameon=False, bbox_to_anchor=(-0.02, 1.0))
fig.savefig(f'{F}/teaser_results.png', dpi=200, facecolor='white')
results = Image.open(f'{F}/teaser_results.png').convert('RGBA').resize((1150, H))

out = Image.new('RGBA', (schema.width + 20 + results.width, H), 'white')
out.alpha_composite(schema, (0, 0))
out.alpha_composite(results, (schema.width + 20, 0))
out.convert('RGB').save(f'{F}/teaser.png')
print(out.size)
