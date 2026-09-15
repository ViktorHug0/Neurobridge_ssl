"""Reproduce MindEye2's published NSD retrieval numbers from extracted clipvoxels.

This is the gate for the whole fMRI experiment: unless the baseline reproduces, any delta
SAGE-TTA appears to buy is meaningless. Nothing here is our method -- it is their evaluation,
re-implemented from final_evaluations.ipynb, so run it and compare against the paper before
touching alignment.

The gallery encoder is rebuilt on open_clip rather than imported from the vendored sgm tree
(whose package __init__ pulls DiffusionEngine -> pytorch_lightning). Two details of
FrozenOpenCLIPImageEmbedder have to be carried over verbatim or the gallery silently differs:

  * preprocess() does `x = (x + 1.0) / 2.0` after resizing, i.e. it expects [-1,1] input.
    all_images.pt is stored in [0,1], so the pipeline actually feeds [0.5,1.0] into the CLIP
    normalisation. That looks like an upstream bug, but training and evaluation share the
    embedder, so it is self-consistent and reproducing their number requires reproducing it.
  * only_tokens=True returns the 256 patch tokens, not the pooled embedding, and
    l2_norm_tokens is left False by both notebooks.

Usage:
  python reproduce_retrieval.py --subj 1 --model_name final_subj01_pretrained_1sess_24bs
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def encode_gallery(images, device, batch_size=8):
    """OpenCLIP ViT-bigG/14 patch tokens, matching FrozenOpenCLIPImageEmbedder exactly."""
    import kornia
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device="cpu")
    del model.transformer                      # text tower is unused, as upstream also does
    visual = model.visual.to(device).half().eval()
    visual.output_tokens = True

    mean = torch.tensor(CLIP_MEAN, device=device)
    std = torch.tensor(CLIP_STD, device=device)

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="gallery"):
            x = images[i:i + batch_size].to(device).float()
            x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic",
                                       align_corners=True, antialias=True)
            x = (x + 1.0) / 2.0                # upstream convention, see module docstring
            x = kornia.enhance.normalize(x, mean, std)
            _, tokens = visual(x.half())
            out.append(tokens.float().cpu())
    return torch.vstack(out)


def batchwise_cosine_similarity(Z, B):
    # verbatim from MindEyeV2/src/utils.py -- note the trailing .T, which decides which axis
    # topk ranks over and therefore which of the two retrieval directions this is
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    return ((Z @ B) / (Z_norm @ B_norm)).T


def topk(similarities, labels, k=1):
    # verbatim from MindEyeV2/src/utils.py
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i + 1)] == labels) / len(labels)
    return topsum.item()


def retrieval_300(clipvoxels, gallery, device, loops=30, pool=300, seed=42):
    """Their protocol: `loops` independent random pools of `pool` candidates, averaged."""
    np.random.seed(seed)
    fwd, bwd = [], []
    for _ in tqdm(range(loops), desc=f"{pool}-way"):
        samp = np.random.choice(np.arange(len(gallery)), size=pool, replace=False)
        emb = nn.functional.normalize(gallery[samp].reshape(pool, -1).to(device), dim=-1)
        emb_ = nn.functional.normalize(clipvoxels[samp].reshape(pool, -1).to(device), dim=-1)
        labels = torch.arange(pool, device=device)
        fwd.append(topk(batchwise_cosine_similarity(emb_, emb), labels))
        bwd.append(topk(batchwise_cosine_similarity(emb, emb_), labels))
    return float(np.mean(fwd)), float(np.mean(bwd))


def retrieval_full(clipvoxels, gallery, device):
    """Full-gallery retrieval: every image is a candidate, one query each.

    SAGE's Sinkhorn step needs a single shared gallery with a balanced assignment, which the
    redrawn 300-pools do not provide. This is the harder task and the one SAGE will be fit on.
    """
    n = len(gallery)
    emb = nn.functional.normalize(gallery.reshape(n, -1).to(device), dim=-1)
    emb_ = nn.functional.normalize(clipvoxels.reshape(n, -1).to(device), dim=-1)
    labels = torch.arange(n, device=device)
    return (topk(batchwise_cosine_similarity(emb_, emb), labels),
            topk(batchwise_cosine_similarity(emb, emb_), labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", default="final_subj01_pretrained_1sess_24bs")
    ap.add_argument("--data_path", default="/nasbrain/p20fores/mindeye_data")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = torch.load(f"{args.data_path}/clipvoxels/clipvoxels_subj{args.subj}_{args.model_name}.pt",
                   weights_only=False)
    clipvoxels = d["clipvoxels"].float()   # older extractions were written in fp16
    images = torch.load(f"{args.data_path}/evals/all_images.pt", weights_only=False)
    assert len(images) == len(clipvoxels), (images.shape, clipvoxels.shape)

    cache = f"{args.data_path}/gallery_bigG_tokens.pt"
    try:
        gallery = torch.load(cache, weights_only=False)
        print(f"gallery from cache {tuple(gallery.shape)}")
    except FileNotFoundError:
        gallery = encode_gallery(images, args.device)
        torch.save(gallery, cache)
        print(f"encoded gallery {tuple(gallery.shape)} -> {cache}")
    assert gallery.shape == clipvoxels.shape, (gallery.shape, clipvoxels.shape)

    fwd, bwd = retrieval_300(clipvoxels, gallery, args.device)
    print(f"\n300-way  image retrieval (fwd) {fwd*100:.1f}%   brain retrieval (bwd) {bwd*100:.1f}%"
          f"   chance {100/300:.1f}%")
    ffwd, fbwd = retrieval_full(clipvoxels, gallery, args.device)
    print(f"{len(gallery)}-way image retrieval (fwd) {ffwd*100:.1f}%   brain retrieval (bwd) "
          f"{fbwd*100:.1f}%   chance {100/len(gallery):.1f}%")


if __name__ == "__main__":
    main()
