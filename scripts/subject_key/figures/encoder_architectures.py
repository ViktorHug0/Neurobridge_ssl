"""Block diagram of EEGProject, TSConv and ATM as nested architectures -> SVG (rendered to PNG with inkscape)."""
import sys
W, GUT, CW, GAP, BW, BH = 1540, 180, 420, 20, 300, 40
CX = [GUT + i * (CW + GAP) + CW / 2 for i in range(3)]
FONT = "Inter, 'Source Sans 3', 'Fira Sans', 'Open Sans', Arial, sans-serif"
MONO = "'DejaVu Sans Mono', 'Fira Mono', monospace"
INK, MUTED = '#1B2430', '#5F6B7A'
FAM = {  # band tint, block fill, stroke
    'att':  ('#F1EFFA', '#D8D1F3', '#5946B0'),
    'conv': ('#EAF5F3', '#C2E4DF', '#1F7A70'),
    'head': ('#FBF4E3', '#F6E3B6', '#A8741A'),
    'io':   ('#FFFFFF', '#EEF1F4', '#6B7785'),
}
o, bg = [], []


def text(x, y, s, size=13.5, fill=INK, anchor='middle', weight=400, family=FONT, extra=''):
    o.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>')


def block(cx, y, fam, name, detail='', h=BH, sub=None, w=BW, rx=7):
    _, fill, stroke = FAM[fam]
    o.append(f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    label = f'<tspan font-weight="600">{name}</tspan>' + (f'<tspan fill="{MUTED}">&#160;&#160;·&#160;&#160;</tspan>{detail}' if detail else '')
    text(cx, y + (h / 2 + 5 if not sub else 22), label)
    if sub:
        text(cx, y + 41, sub, size=11.5, fill=MUTED)
    return y + h


def arrow(x, y1, y2, label=None, ly=None):
    o.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 2}" stroke="{INK}" stroke-width="1.5" marker-end="url(#a)"/>')
    if label:
        text(x + 11, (ly if ly is not None else (y1 + y2) / 2) + 4, label, size=11, fill=MUTED, anchor='start', family=MONO)


def skip(cx, y_tap, y_into):
    xl = cx - BW / 2 - 20
    o.append(f'<circle cx="{cx}" cy="{y_tap}" r="2.6" fill="{INK}"/>')
    o.append(f'<path d="M{cx} {y_tap} H{xl} V{y_into} H{cx - BW / 2 - 2}" fill="none" stroke="{INK}" stroke-width="1.5" marker-end="url(#a)"/>')


def band(top, bottom, fam, title, who):
    tint, _, stroke = FAM[fam]
    bg.append(f'<rect x="20" y="{top}" width="{W - 40}" height="{bottom - top}" rx="12" fill="{tint}"/>')
    mid = (top + bottom) / 2
    for i, line in enumerate(title):
        text(38, mid - 8 * (len(title) - 1) + 19 * i - 6, line, size=15, fill=stroke, anchor='start', weight=700)
    text(38, mid + 8 * (len(title) - 1) + 16, who, size=12, fill=MUTED, anchor='start')


def group(cx, top, bottom, fam, caption, new):
    stroke = FAM[fam][2]
    dash = '' if new else 'stroke-dasharray="5 4"'
    bg.append(f'<rect x="{cx - 190}" y="{top + 8}" width="380" height="{bottom - top - 16}" rx="10" fill="#FFFFFF" fill-opacity="{0.75 if new else 0.5}" stroke="{stroke}" stroke-width="{1.6 if new else 1.1}" {dash}/>')
    text(cx - 176, top + 27, caption.upper(), size=10.5, fill=stroke if new else MUTED, anchor='start', weight=700, extra='letter-spacing="0.9"')


def absent(cx, top, bottom):
    text(cx + 12, (top + bottom) / 2 + 4, 'not present', size=11.5, fill=MUTED, anchor='start', extra='font-style="italic"')


# ---- rows
P = 68  # pitch of 40-px blocks
y_in = 100
A_top = 168; a = [A_top + 38]; a.append(a[0] + 54 + 28); a += [a[1] + P * i for i in range(1, 5)]; A_bot = a[-1] + BH + 18
B_top = A_bot + 18; b = [B_top + 38 + P * i for i in range(6)]; B_bot = b[-1] + BH + 18
y_fl = B_bot + 22
C_top = y_fl + 34 + 22; c = [C_top + 38 + P * i for i in range(3)]; C_bot = c[-1] + BH + 18
y_pr = C_bot + 26; y_out = y_pr + BH + 26; H = y_out + 38 + 62

o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="EEGProject, TSConv and ATM as nested encoder architectures">')
o.append(f'<defs><marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{INK}"/></marker></defs>')
o.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
BG_AT = len(o)

# ---- titles and "builds on" arrows
for cx, name, sub in zip(CX, ('EEGProject', 'TSConv', 'ATM'), ('MLP on the flattened epoch', 'EEGProject preceded by a convolution stem', 'TSConv preceded by an electrode transformer')):
    text(cx, 50, name, size=23, weight=700)
    text(cx, 74, sub, size=13, fill=MUTED)
for i, (lab, l, r) in enumerate((('+ temporal–spatial convolution', 82, 58), ('+ attention across electrodes', 58, 40))):
    o.append(f'<line x1="{CX[i] + l}" y1="42" x2="{CX[i + 1] - r - 2}" y2="42" stroke="{INK}" stroke-width="1.5" marker-end="url(#a)"/>')
    text((CX[i] + l + CX[i + 1] - r) / 2, 33, lab, size=12, fill=INK, weight=600)

# ---- bands
band(A_top, A_bot, 'att', ['Electrode', 'transformer'], 'ATM only')
band(B_top, B_bot, 'conv', ['Temporal–', 'spatial', 'convolution'], 'TSConv, ATM')
band(C_top, C_bot, 'head', ['Projection', 'head'], 'all three')

for i, cx in enumerate(CX):
    block(cx, y_in, 'io', 'EEG epoch', '63 electrodes × 250 samples', h=44)
    # --- electrode transformer
    if i == 2:
        group(cx, A_top, A_bot, 'att', 'added by ATM', True)
        arrow(cx, y_in + 44, a[0], '63 × 250', ly=y_in + 44 + 12)
        block(cx, a[0], 'att', 'Token embedding', 'Linear 250 → 250', h=54, sub='one token per electrode, plus one shared token')
        arrow(cx, a[0] + 54, a[1], '64 tokens × 250', ly=a[0] + 54 + 7)
        block(cx, a[1], 'att', 'Self-attention across electrodes', '4 heads')
        arrow(cx, a[1] + BH, a[2])
        block(cx, a[2], 'att', 'Add &amp; LayerNorm')
        arrow(cx, a[2] + BH, a[3])
        block(cx, a[3], 'att', 'Feed-forward', '250 → 256 → 250, GELU')
        arrow(cx, a[3] + BH, a[4])
        block(cx, a[4], 'att', 'Add &amp; LayerNorm')
        arrow(cx, a[4] + BH, a[5])
        block(cx, a[5], 'att', 'LayerNorm')
        skip(cx, a[1] - 16, a[2] + BH / 2)
        skip(cx, a[3] - 16, a[4] + BH / 2)
        arrow(cx, a[5] + BH, b[0], '63 × 250  (first 63 tokens kept)', ly=A_bot + 5)
    else:
        absent(cx, A_top, A_bot)
    # --- temporal-spatial convolution
    if i >= 1:
        group(cx, B_top, B_bot, 'conv', 'added by TSConv' if i == 1 else 'same as TSConv', i == 1)
        if i == 1:
            arrow(cx, y_in + 44, b[0], '63 × 250', ly=y_in + 44 + 12)
        block(cx, b[0], 'conv', 'Temporal convolution', '40 filters, 1 × 25')
        arrow(cx, b[0] + BH, b[1], '40 × 63 × 226')
        block(cx, b[1], 'conv', 'Average pooling', '1 × 51, stride 5')
        arrow(cx, b[1] + BH, b[2], '40 × 63 × 36')
        block(cx, b[2], 'conv', 'BatchNorm', 'ELU')
        arrow(cx, b[2] + BH, b[3])
        block(cx, b[3], 'conv', 'Spatial convolution', '40 filters, 63 × 1')
        arrow(cx, b[3] + BH, b[4], '40 × 1 × 36')
        block(cx, b[4], 'conv', 'BatchNorm', 'ELU · Dropout 0.5')
        arrow(cx, b[4] + BH, b[5])
        block(cx, b[5], 'conv', '1 × 1 convolution', '40 filters')
        arrow(cx, b[5] + BH, y_fl, '40 × 1 × 36', ly=B_bot + 7)
    else:
        absent(cx, B_top, B_bot)
        arrow(cx, y_in + 44, y_fl, '63 × 250', ly=y_in + 44 + 12)
    # --- flatten, head, projector, output
    n = '15,750' if i == 0 else '1,440'
    block(cx, y_fl, 'io', 'Flatten', h=34, w=130, rx=17)
    group(cx, C_top, C_bot, 'head', 'base model' if i == 0 else 'same as EEGProject', i == 0)
    arrow(cx, y_fl + 34, c[0], n, ly=y_fl + 34 + 12)
    block(cx, c[0], 'head', 'Linear', f'{n} → D')
    arrow(cx, c[0] + BH, c[1], 'D', ly=c[0] + BH + 7)
    block(cx, c[1], 'head', 'GELU', f'Linear D → D · Dropout {0.3 if i == 0 else 0.5}')
    arrow(cx, c[1] + BH, c[2])
    block(cx, c[2], 'head', 'Add &amp; LayerNorm')
    skip(cx, c[1] - 16, c[2] + BH / 2)
    arrow(cx, c[2] + BH, y_pr, 'D', ly=C_bot + 9)
    block(cx, y_pr, 'io', 'Linear projector', 'D → d')
    arrow(cx, y_pr + BH, y_out, 'd')
    block(cx, y_out, 'io', 'EEG embedding', 'd dimensions', h=38, rx=19)

text(38, H - 34, 'Shapes are channels × electrodes × time for one epoch.  D: width of the head;  d: size of the shared EEG–image embedding.', size=12, fill=MUTED, anchor='start')
text(38, H - 15, 'Dashed frames: blocks reused from an earlier model.  Left-hand loops: residual connections.', size=12, fill=MUTED, anchor='start')
o[BG_AT:BG_AT] = bg
o.append('</svg>')
open(sys.argv[1], 'w').write('\n'.join(o))
