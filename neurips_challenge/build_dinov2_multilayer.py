"""Extra DINOv2-giant image targets for multi-level / blur alignment (ALIGN_LOG.md).
Same model and preprocessing as the official Track-1 targets: neuralset resizes to 518, then the HF DINOv2
processor resizes the short side to 256 and centre-crops 224 (it cancels imsize); hidden state int(.6667 * 40) =
26, mean over tokens. `--check` verifies the match with dinov2-giant_track1 before anything is written.
Outputs data/things_eeg/image_feature/dinov2-giant_L{l}_mean/{image_train,image_test}.npy for each layer, and
dinov2-giant_track1_blur{s}/... = the track1 layer on Gaussian-blurred images (sigma s px at 518).
    build_dinov2_multilayer.py --check
    build_dinov2_multilayer.py --layers 8 16 34 40 --blur 6 12
"""
import argparse, os
import numpy as np
import torch
from PIL import Image, ImageFilter

IMAGE_SET = '/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set'
OUT = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature'
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def images(split):
    d = f'{IMAGE_SET}/{split}_images'
    return [os.path.join(d, c, f) for c in sorted(os.listdir(d)) for f in sorted(os.listdir(os.path.join(d, c)))]


def prep(path, blur=0):
    im = Image.open(path).convert('RGB')
    im = im.resize((518, round(518 * im.height / im.width)) if im.width <= im.height else (round(518 * im.width / im.height), 518), Image.BILINEAR)
    if blur:  # sigma in pixels of the 518 image (the viewed size), before the processor's downsampling
        im = im.filter(ImageFilter.GaussianBlur(blur))
    w, h = im.size; r = 256 / min(w, h)
    im = im.resize((round(w * r), round(h * r)), Image.BICUBIC)
    w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    return torch.tensor(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1), dtype=torch.float32)


@torch.no_grad()
def run(model, paths, blur=0, bs=8, states=None):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([prep(p, blur) for p in paths[i:i + bs]]).cuda().half()
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states   # 41: embeddings + 40 blocks
        keep = range(len(hs)) if states is None else states                    # only the needed states stay in RAM
        out.append(torch.stack([hs[j].float().mean(1) for j in keep], 1).cpu())  # (b, len(keep), 1536) mean over tokens
        if i % 800 == 0:
            print(f'{i}/{len(paths)}', flush=True)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true')
    ap.add_argument('--layers', type=int, nargs='*', default=[]); ap.add_argument('--blur', type=float, nargs='*', default=[])
    ap.add_argument('--track1_state', type=int, default=26, help='hidden-state index of the official target (from --check)')
    a = ap.parse_args()
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()
    ref = np.load(f'{OUT}/dinov2-giant_track1/image_train.npy').reshape(-1, 1536)
    tr = images('training')
    if a.check:
        H = run(model, tr[:16]).numpy()
        r = ref[:16] / np.linalg.norm(ref[:16], axis=1, keepdims=True)
        for l in range(H.shape[1]):
            h = H[:, l] / np.linalg.norm(H[:, l], axis=1, keepdims=True)
            print(f'hidden state {l:2d}: cosine to track1 target {(h * r).sum(1).mean():.4f}')
        return
    for split, paths, shape in (('train', tr, (1654, 10)), ('test', images('test'), (200, 1))):
        if a.layers:
            H = run(model, paths, states=a.layers)
            for j, l in enumerate(a.layers):
                d = f'{OUT}/dinov2-giant_L{l}_mean'; os.makedirs(d, exist_ok=True)
                np.save(f'{d}/image_{split}.npy', H[:, j].numpy().reshape(*shape, 1536))
            del H
        for s in a.blur:
            Hb = run(model, paths, blur=s, states=[a.track1_state])[:, 0].numpy().reshape(*shape, 1536)
            d = f'{OUT}/dinov2-giant_track1_blur{s:g}'; os.makedirs(d, exist_ok=True)
            np.save(f'{d}/image_{split}.npy', Hb)
        print(f'{split} done', flush=True)


if __name__ == '__main__':
    main()
