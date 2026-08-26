"""Gate for the CLIP-space caption route: does the EEG survive a hop into raw CLIP ViT-H/14?

The Q-Former bridge failed because 512 -> 32x768 is a huge expansion into a space where a partly
wrong prefix makes OPT hallucinate. CLIP ViT-H/14 is 1024-d and semantic, so the same hop should be
far easier -- but only if the EEG signal actually survives it. Fit the map on IMAGE pairs (both
feature sets exist for the same images), then push EEG through and compare retrieval to the 81%
the EEG gets in its own shared space. If that collapses, no captioner downstream can rescue it.

  .venv/bin/python clip_space_probe.py
"""
import numpy as np
import torch
import torch.nn.functional as F

import caption_bridge as cb

CKPT = "results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01"
CLIP = "data/things_eeg/image_feature/ViT-H-14_final"


class Args:
    input, checkpoint_dir, feature_dim = "proj", CKPT, 512


def clip_feats(split):
    x = np.load(f"{CLIP}/image_{'train' if split == 'training' else 'test'}.npy")
    return F.normalize(torch.from_numpy(x.reshape(-1, x.shape[-1])).float(), dim=-1)


def main():
    xtr, xte = cb.embeddings("training", Args), cb.embeddings("test", Args)
    ytr, yte = clip_feats("training"), clip_feats("test")
    print(f"InternViT shared {tuple(xtr.shape)} -> CLIP ViT-H/14 {tuple(ytr.shape)}")

    # ridge in closed form: 512x512 solve, no iteration needed
    lam = 1e-3 * len(xtr)
    A = xtr.T @ xtr + lam * torch.eye(xtr.shape[1])
    W = torch.linalg.solve(A, xtr.T @ ytr)

    def top1(pred, bank):
        pred = F.normalize(pred, dim=-1)
        return 100 * (pred @ bank.T).argmax(1).eq(torch.arange(len(pred))).float().mean()

    pte = xte @ W
    r2 = 1 - (pte - yte).pow(2).sum() / (yte - ytr.mean(0)).pow(2).sum()
    print(f"\nimage -> CLIP (the map's own ceiling)")
    print(f"  R^2 on held-out test images     : {r2:.4f}")
    print(f"  200-way top1 in CLIP space      : {top1(pte, yte):.2f}%")

    d = np.load("data/things_eeg/captions/eeg_sub01.npz")
    e = F.normalize(torch.from_numpy(d["eeg"]).float(), dim=-1)
    i_shared = F.normalize(torch.from_numpy(d["image"]).float(), dim=-1)
    print(f"\nEEG")
    print(f"  200-way top1 in the shared space: {top1(e, i_shared):.2f}%   <- what we start with")
    print(f"  200-way top1 in CLIP space      : {top1(e @ W, yte):.2f}%   <- after the hop")
    # a shifted EEG cloud is the cheap domain-adaptation variant, free to test here
    es = F.normalize(e - e.mean(0) + i_shared.mean(0), dim=-1)
    print(f"  ... mean-matched first          : {top1(es @ W, yte):.2f}%")


if __name__ == "__main__":
    main()
