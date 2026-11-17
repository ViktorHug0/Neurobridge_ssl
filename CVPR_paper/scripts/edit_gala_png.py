"""GALA diagram for the final method (2026-11-17): edits a copy of figures/GALA.png (no editable source exists).
  - Sinkhorn panel: the uniform column marginals "1 1 1 1" become estimated masses b_1 ... b_4 (rows keep mass 1)
  - the canvas grows downwards by a full-width panel "4) Refine alignment with new queries": queries arrive along a
    time axis (thumbnails = crops of the traces of panel 1), the new query (blue frame) feeds the CSLS -> Sinkhorn ->
    Procrustes cycle (arrow up through the gap between the CSLS and Procrustes boxes), future queries are dashed, and
    the weight of the buffer against the zero-shot prior, alpha_t = t / (t + n0), rises under the axis
Writes figures/GALA_v2.png.  usage: edit_gala_png.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

F = 'CVPR_paper/figures/'
LS = '/usr/share/fonts/truetype/liberation/'
WHITE, INK, NAVY, BLUE, GREY = (255, 255, 255, 255), (0, 0, 0, 255), (4, 17, 62, 255), (42, 120, 214, 255), (150, 150, 150, 255)
src = Image.open(F + 'GALA.png').convert('RGBA')

# canvas grows by DY rows: content rows 0-1229 unchanged; the outer frame starts curving into its bottom corners at
# y = 1218, so its sides (x < 70, x >= 1756) are continued straight from row 1200 and its bottom (corners from y = 1215,
# straight edge from y = 1230) is moved down by DY; the new rows are background
DY, CUT, L, R = 370, 1230, 70, 1756
a = np.array(src)
h = len(a)
out = np.empty((h + DY, a.shape[1], 4), a.dtype)
out[:CUT] = a[:CUT]
out[1215:CUT + DY, :L], out[1215:CUT + DY, R:] = a[1200, :L], a[1200, R:]
out[CUT:CUT + DY, L:R] = (250, 224, 206, 255)
out[1215 + DY:, :L], out[1215 + DY:, R:] = a[1215:, :L], a[1215:, R:]
out[CUT + DY:, L:R] = a[CUT:, L:R]
im = Image.fromarray(out)
d = ImageDraw.Draw(im)

# 1. column marginals: erase "1 1 1 1" (bbox x 1481-1661, y 480-513) and write b_1 .. b_4 over the four columns
d.rectangle((1470, 470, 1690, 522), fill=WHITE)
fb = ImageFont.truetype(LS + 'LiberationSerif-BoldItalic.ttf', 50)
fs = ImageFont.truetype(LS + 'LiberationSerif-Bold.ttf', 28)
for j, cx in enumerate((1489, 1543, 1597, 1650)):
    w = d.textlength('b', font=fb) + d.textlength(str(j + 1), font=fs)
    x0 = cx - w / 2
    d.text((x0, 507), 'b', font=fb, fill=INK, anchor='ls')
    d.text((x0 + d.textlength('b', font=fb), 515), str(j + 1), font=fs, fill=INK, anchor='ls')

# 2. panel 4, aligned with panel 3 (left edge x = 83) and panel 2 (right edge x = 1753)
box = (83, 1246, 1753, 1588)
d.rounded_rectangle(box, radius=24, fill=WHITE, outline=NAVY, width=2)
d.text((box[0] + 22, 1308), '4) Refine alignment with new queries', font=ImageFont.truetype(LS + 'LiberationSerif-Bold.ttf', 52),
       fill=INK, anchor='ls')  # left-aligned (user)
traces = [src.crop((433, 243, 564, 364)), src.crop((695, 258, 826, 356))]  # interiors only: the frames are redrawn
W, H, TOP = 104, 94, 1336
XNEW = 1349 - W // 2  # the new query sits under the gap between the CSLS and Procrustes boxes
past = [200 + i * 122 for i in range(8)]
for i, x in enumerate(past + [XNEW]):
    im.paste(traces[i % 2].resize((W, H), Image.LANCZOS), (x, TOP))
    d.rectangle((x - 1, TOP - 1, x + W, TOP + H), outline=(40, 40, 40, 255), width=2)
d.rectangle((XNEW - 5, TOP - 5, XNEW + W + 4, TOP + H + 4), outline=BLUE, width=6)
d.text(((past[-1] + W + XNEW) / 2, TOP + H / 2 + 12), '...', font=ImageFont.truetype(LS + 'LiberationSerif-Bold.ttf', 56), fill=INK, anchor='ms')
for x in (XNEW + 150, XNEW + 272):  # queries still to come: dashed frames
    for (xa, ya, xb, yb) in ((x, TOP, x + W, TOP), (x, TOP + H, x + W, TOP + H), (x, TOP, x, TOP + H), (x + W, TOP, x + W, TOP + H)):
        n = int(max(abs(xb - xa), abs(yb - ya)) // 14)
        for s in range(0, n, 2):
            d.line((xa + (xb - xa) * s / n, ya + (yb - ya) * s / n, xa + (xb - xa) * (s + 1) / n, ya + (yb - ya) * (s + 1) / n), fill=GREY, width=3)

# time axis and alpha_t = t / (t + n0)
X0, X1, YB, YT = 200, 1690, 1556, 1452
t = np.linspace(0, 1, 300)
at = t * 5 / (t * 5 + 1)
d.line(list(zip(X0 + t * (X1 - X0), YB - at * (YB - YT))), fill=BLUE, width=7)
d.line((X0, YB, X1 + 10, YB), fill=INK, width=4)
d.polygon(((X1 + 30, YB), (X1 + 8, YB - 11), (X1 + 8, YB + 11)), fill=INK)
fi = ImageFont.truetype(LS + 'LiberationSerif-BoldItalic.ttf', 50)
fsub = ImageFont.truetype(LS + 'LiberationSerif-BoldItalic.ttf', 32)
fa, fasub = ImageFont.truetype(LS + 'LiberationSerif-BoldItalic.ttf', 75), ImageFont.truetype(LS + 'LiberationSerif-BoldItalic.ttf', 48)  # alpha_t 1.5x (user)
d.text((X1 + 36, YB + 14), 't', font=fi, fill=INK, anchor='ls')
d.text((104, 1514), 'α', font=fa, fill=BLUE, anchor='ls')
d.text((104 + d.textlength('α', font=fa), 1532), 't', font=fasub, fill=BLUE, anchor='ls')
# the current time: dashed line from the new query down to the axis, dot on the curve
xc = 1349
tc = (xc - X0) / (X1 - X0)
yc = YB - tc * 5 / (tc * 5 + 1) * (YB - YT)
for y in range(TOP + H + 8, YB, 16):
    d.line((xc, y, xc, min(y + 8, YB)), fill=BLUE, width=3)
d.ellipse((xc - 10, yc - 10, xc + 10, yc + 10), fill=BLUE)

# arrow from the new query up into the cycle, through the gap between the CSLS and Procrustes boxes
d.line((xc, TOP - 8, xc, 945), fill=INK, width=8)
d.polygon(((xc, 920), (xc - 15, 948), (xc + 15, 948)), fill=INK)
im.save(F + 'GALA_v2.png')
