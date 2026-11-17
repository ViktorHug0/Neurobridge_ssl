"""Q01 the 10,000 stimulus images of MSS (OpenNeuro ds005589: 32 people, 5 Hz RSVP, 100 ms on / 100 ms off) and their grader targets.
The dataset ships file names only (stimuli/<class>.csv: name, source, class). Sources on this cluster: ImageNet per-class
tars (/nasbrain/datasets/imagenet/train/<wnid>.tar), VOC2012 JPEGImages, and VOC2007 (downloaded tars, 6-digit names).
Images are copied to <D>/images/<name>.jpg; targets = DINOv2-giant hidden state 26, token mean, HF processor (resize 256,
centre crop 224), as logs/things_distractors.py (checked there against the grader cache: cos .999). The participants saw the
image at height 640 with its aspect ratio kept; the target is the centre crop.
-> <D>/targets.npz: names (10000,), cls (10000,), F (10000, 1536) float32
    nq01_mss_images.py [--gather-only]"""
import csv, glob, io, os, sys, tarfile
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np

D = '/nasbrain/p20fores/mss_ds005589'; IMG = f'{D}/images'; os.makedirs(IMG, exist_ok=True)
CLASSES = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/meta/*.csv') if os.path.basename(f)[0].islower())
rows = [(r['image_file_name'], r['dataset'], c) for c in CLASSES for r in csv.DictReader(open(f'{D}/meta/{c}.csv'))]
assert len(CLASSES) == 20 and len(rows) == 10000 and len({n for n, _, _ in rows}) == 10000, (len(CLASSES), len(rows))
todo = {n: d for n, d, _ in rows if not os.path.exists(f'{IMG}/{n}.jpg')}


def from_tar(path, wanted):  # wanted: {member basename without extension: output name}; members are written under our own names only
    with tarfile.open(path) as t:
        for m in t:
            b = os.path.splitext(os.path.basename(m.name))[0]
            if m.isfile() and b in wanted and m.name.lower().endswith(('.jpg', '.jpeg')):
                open(f'{IMG}/{wanted[b]}.jpg', 'wb').write(t.extractfile(m).read())


by_wnid = {}
for n, d in todo.items():
    if d == 'ImageNet':
        by_wnid.setdefault(n.split('_')[0], {})[n] = n
    elif '_' in n:
        open(f'{IMG}/{n}.jpg', 'wb').write(open(f'/nasbrain/datasets/VOC2012/JPEGImages/{n}.jpg', 'rb').read())
for w, wanted in by_wnid.items():
    from_tar(f'/nasbrain/datasets/imagenet/train/{w}.tar', wanted); print('ImageNet', w, len(wanted), flush=True)
v7 = {n: n for n, d in todo.items() if d == 'PASCAL' and '_' not in n}
if v7:
    for t in glob.glob(f'{D}/voc2007/*.tar'):
        from_tar(t, v7)
missing = [n for n, _, _ in rows if not os.path.exists(f'{IMG}/{n}.jpg')]
print('images present %d of 10000; missing e.g. %s' % (10000 - len(missing), missing[:5]), flush=True)
if '--gather-only' in sys.argv or missing:
    sys.exit()
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import to_tensor
dev = torch.device('cuda:0'); proc = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', do_rescale=False)
model = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float32).to(dev).eval(); out = []
for b in range(0, 10000, 16):
    x = proc(images=[to_tensor(Image.open(f'{IMG}/{n}.jpg').convert('RGB')) for n, _, _ in rows[b:b + 16]], return_tensors='pt')['pixel_values'].to(dev)
    with torch.no_grad():
        out.append(model(pixel_values=x, output_hidden_states=True).hidden_states[26].float().mean(1).cpu())
    if b % 1600 == 0:
        print(b, flush=True)
np.savez(f'{D}/targets.npz', names=np.array([n for n, _, _ in rows]), cls=np.array([CLASSES.index(c) for _, _, c in rows]), F=torch.cat(out).numpy().astype(np.float32))
print('saved', flush=True)
