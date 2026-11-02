"""Denoising pretraining for the scratch Squeezeformer: single trial -> mean of the subject's OTHER 3 trials.

The scratch models do well on averaged trials and poorly on single ones; this teaches the trunk the
mapping from a noisy trial onto its clean evoked response, from training EEG only. Optional MAE-style
time masking of the input. Saves a state dict that train.py loads with --init_encoder (strict=False:
the CLIP head is not pretrained).

    pretrain_denoise.py <out.pth> [--epochs 20] [--mask_frac 0.3] [--power_branch]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_experiments.architectures.ortho_encoders import FastTSConvSqueezeformer  # noqa: E402

DATA = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench"


class Denoiser(nn.Module):
    """Squeezeformer trunk (stem + blocks) and a per-token linear decoder back to (63, 120)."""

    def __init__(self, encoder, n_chans=63, n_times=120):
        super().__init__()
        self.encoder = encoder
        with torch.no_grad():
            n_tokens = self.tokens(torch.zeros(2, n_chans, n_times)).shape[1]
        self.samples_per_token = -(-n_times // n_tokens)
        self.decoder = nn.Linear(encoder.token_projection[0].out_features, n_chans * self.samples_per_token)
        self.n_chans, self.n_times = n_chans, n_times

    def tokens(self, x):
        e = self.encoder
        z = e.stem_after_pool(e.temporal(x.unsqueeze(1))).squeeze(2).transpose(1, 2)
        z = e.token_projection(z) + e.position[:, :z.shape[1]]
        for block in e.blocks:
            z = block(z)
        return z

    def forward(self, x):
        z = self.decoder(self.tokens(x))  # (B, T', C * s)
        b, t, _ = z.shape
        z = z.view(b, t, self.n_chans, self.samples_per_token).permute(0, 2, 1, 3).reshape(b, self.n_chans, -1)
        return F.interpolate(z, size=self.n_times, mode="linear", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_frac", type=float, default=0.0, help="zero this fraction of time samples (contiguous blocks of 12) in the input")
    parser.add_argument("--power_branch", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")

    trials, averages = [], []
    for s in range(1, 11):
        x = np.load(f"{DATA}/sub-{s:02d}/train.npy", mmap_mode="r")  # (1654, 10, 4, 63, 120)
        x = np.asarray(x, dtype=np.float16).reshape(-1, 4, 63, 120)
        trials.append(x)
        averages.append(x.astype(np.float32).mean(1).astype(np.float16))
        print(f"loaded sub-{s:02d}", flush=True)
    trials, averages = np.concatenate(trials), np.concatenate(averages)  # (N, 4, C, T), (N, C, T)
    n = len(trials)

    encoder = FastTSConvSqueezeformer(1536, 120, 63, pool_kernel=25, pool_stride=2, backbone_dim=512,
                                      power_branch=args.power_branch)
    model = Denoiser(encoder).to(device)
    params = [p for name, p in model.named_parameters() if not name.startswith("encoder.head")
              and not name.startswith("encoder.power")]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (n * 4 // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=0.1)
    print(f"{n * 4} trials, {steps} steps", flush=True)

    step = 0
    for epoch in range(args.epochs):
        order = np.random.permutation(n * 4)
        t0, total = time.time(), 0.0
        for i in range(n * 4 // args.batch):
            idx = order[i * args.batch:(i + 1) * args.batch]
            x = torch.from_numpy(trials[idx // 4, idx % 4].astype(np.float32)).to(device)
            # Leave-one-out target: the mean of the other three repetitions, so no part of the
            # target is a copy of the input.
            y = (4 * torch.from_numpy(averages[idx // 4].astype(np.float32)).to(device) - x) / 3
            if args.mask_frac > 0:
                keep = torch.ones(x.shape[0], 1, 120 // 12, device=device)
                keep[torch.rand_like(keep) < args.mask_frac] = 0
                x = x * keep.repeat_interleave(12, dim=-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            step += 1
        print(f"epoch {epoch + 1}: mse {total / (i + 1):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({k: v for k, v in encoder.state_dict().items()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
