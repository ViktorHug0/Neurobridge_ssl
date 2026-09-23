"""Neurobridge image-augmentation targets in the Track-1 DINOv2 space.

Replicates extract_feature.py + fuse_feature.py -- GaussianBlur 15, GaussianNoise 25,
LowResolution 0.1, Mosaic 5, then the mean of the four per image, which is what all 70
`--image_aug` runs in this repo trained on -- but embeds every augmented image with
neuralset's own HuggingFaceImage (facebook/dinov2-giant, imsize 518, layers 2/3, token mean),
the extractor that produced the grader's targets. The augmented targets therefore live in
exactly the space Codabench ranks in.

    build_aug_targets.py --verify 500   # clean images must reproduce dinov2-giant_track1
    build_aug_targets.py                # 4 augs -> <out>/<aug>/, fused -> <out>/<a-b-c-d>/

Run with the challenge venv (neuralset, transformers 4.57, the versions behind the official
targets). Only cv2, which the augmentations need, is borrowed from this repo's venv.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# Appended, not prepended: the challenge venv's numpy/torch/transformers must win.
sys.path.append(os.path.join(REPO, ".venv/lib/python3.12/site-packages"))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from neuralset.extractors.image import HuggingFaceImage

from module.image_augmentation import GaussianBlur, GaussianNoise, LowResolution, Mosaic
from neurips_challenge.build_dinov2_targets import DIM, IMAGE_SET_DIR, list_images

# extract_feature.py's parameters for each --aug_type.
AUGS = {
    "GaussianBlur": GaussianBlur(blur_kernel_size=15, fluctuation_range=0),
    "GaussianNoise": GaussianNoise(mean=0.0, std=25.0, fluctuation_range=0),
    "LowResolution": LowResolution(scale=0.1),
    "Mosaic": Mosaic(mosaic_level=5),
}
SPLITS = {  # name -> (image_set subdir, images per object, official target file)
    "train": ("training_images", 10, "image_train.npy"),
    "test": ("test_images", 1, "image_test.npy"),
}
FEATURES = os.path.join(REPO, "data/things_eeg/image_feature")
OFFICIAL = os.path.join(FEATURES, "dinov2-giant_track1")


def split_paths(split: str) -> list[str]:
    image_dir = os.path.join(IMAGE_SET_DIR, SPLITS[split][0])
    return [os.path.join(image_dir, c, f) for c, f in list_images(image_dir)]


@torch.no_grad()
def embed(ext, paths, aug, batch_size=32) -> np.ndarray:
    # neuralset's _iter_image_latents: Resize(imsize) -> ToTensor -> processor, then token
    # mean and layer selection per image.
    tf = transforms.Compose([transforms.Resize(ext.imsize), transforms.ToTensor()])
    out = np.empty((len(paths), DIM), dtype=np.float32)
    for start in range(0, len(paths), batch_size):
        imgs = []
        for path in paths[start : start + batch_size]:
            img = Image.open(path).convert("RGB")
            imgs.append(tf(aug(img) if aug is not None else img))
        if all(i.shape == imgs[0].shape for i in imgs):
            batch = torch.stack(imgs).to(ext.model_device)
        else:  # neuralset's collate_fn falls back to a list for mixed sizes
            batch = [i.to(ext.model_device) for i in imgs]
        for j, latent in enumerate(ext._extract_batched_latents(batch)):
            latent = ext._aggregate_layers(ext._aggregate_tokens(latent))
            out[start + j] = latent.float().cpu().numpy()
        if (start // batch_size) % 50 == 0:
            print(f"  {start + len(imgs)}/{len(paths)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=int, default=0, help="embed N clean images per split and compare")
    parser.add_argument("--out_dir", default=os.path.join(FEATURES, "dinov2-giant_track1_aug"))
    parser.add_argument("--batch_size", type=int, default=16, help="16 fits DINOv2-giant fp32 on a 10 GB 3080")
    args = parser.parse_args()

    ext = HuggingFaceImage(model_name="facebook/dinov2-giant", imsize=518, device="cuda")

    if args.verify:
        for split in SPLITS:
            paths = split_paths(split)[: args.verify]
            got = embed(ext, paths, None, args.batch_size)
            ref = np.load(os.path.join(OFFICIAL, SPLITS[split][2])).reshape(-1, DIM)[: len(paths)]
            cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
            rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
            print(f"{split}: n={len(paths)} cosine min {cos.min():.6f} mean {cos.mean():.6f} "
                  f"| rel err max {rel.max():.2e}")
            assert cos.min() > 0.999, "clean re-extraction does not reproduce the official targets"
        return

    fused = {split: [] for split in SPLITS}
    for name, aug in AUGS.items():
        np.random.seed(0)  # GaussianNoise draws from np.random: fixed so targets are reproducible
        for split, (_, per_object, _) in SPLITS.items():
            path = os.path.join(args.out_dir, name, f"{split}.npy")
            if os.path.exists(path):
                feats = np.load(path)
            else:
                print(f"{name}/{split}", flush=True)
                feats = embed(ext, split_paths(split), aug, args.batch_size).reshape(1, -1, per_object, DIM)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, feats)
            fused[split].append(feats)
    # fuse_feature.py: mean over the augmentations, keepdims -> (1, objects, images, DIM).
    fused_dir = os.path.join(args.out_dir, "-".join(AUGS))
    os.makedirs(fused_dir, exist_ok=True)
    for split, parts in fused.items():
        mean = np.mean(np.concatenate(parts, axis=0), axis=0, keepdims=True)
        np.save(os.path.join(fused_dir, f"{split}.npy"), mean)
        print(f"fused {split}: {mean.shape} -> {fused_dir}")


if __name__ == "__main__":
    main()
