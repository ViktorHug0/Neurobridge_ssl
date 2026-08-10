"""Phase 1b prep: encode the (blurred) stimulus images into SDXL VAE latents.

These are the regression targets for the low-level decoder (EEG -> coarse image). We blur the
stimulus first so the target carries low-level layout/colour, not high-level detail (ATM/MindEye
low-level pipeline). Output: latents (N_obj, N_img, 4, 64, 64) fp16, matching the image_feature order.
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import AutoencoderKL
from tqdm import tqdm

IMG_SET = "/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set"


def stim_paths(split):
    sub = "test_images" if split == "test" else "training_images"
    dirs = sorted(glob.glob(os.path.join(IMG_SET, sub, "*/")))
    # (n_objects, n_images_per_object) grid, matching extract_feature ordering
    return [sorted(glob.glob(os.path.join(d, "*"))) for d in dirs]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", default="data/things_eeg/vae_latents")
    ap.add_argument("--res", type=int, default=512)     # SDXL latent = res/8
    ap.add_argument("--blur", type=float, default=6.0)  # gaussian blur radius for low-level target
    ap.add_argument("--hf_cache", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    # fp32: the SDXL VAE overflows to NaN in fp16 (hence sdxl-vae-fp16-fix). Encode in fp32.
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32,
                                        cache_dir=args.hf_cache).to(device).eval()
    grid = stim_paths(args.split)
    n_obj, n_img = len(grid), len(grid[0])
    lat = np.zeros((n_obj, n_img, 4, args.res // 8, args.res // 8), np.float16)
    for oi, imgs in enumerate(tqdm(grid, desc=f"vae {args.split}")):
        for ii, p in enumerate(imgs):
            im = Image.open(p).convert("RGB").resize((args.res, args.res)).filter(
                ImageFilter.GaussianBlur(args.blur))
            x = (torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0)
                 .permute(2, 0, 1).unsqueeze(0).float().to(device))
            z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            lat[oi, ii] = z.squeeze(0).cpu().numpy()
    np.save(os.path.join(args.out, f"{args.split}.npy"), lat)
    print(f"saved {args.split}: {lat.shape}  scaling={vae.config.scaling_factor}")


if __name__ == "__main__":
    main()
