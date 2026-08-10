"""JMVR reconstruction metric suite.

Reproduces the metrics reported in the JMVR / high-fidelity-recon paper (Table 1):
  low/pixel : PixCorr (up), SSIM (up)
  novel     : LabEMD (down, CIELab colour EMD), DepthEMD (down, Depth-Anything-v2 depth EMD)
  semantic  : AlexNet(2)/(5), Inception, CLIP (2-way identification, up), SwAV (down, corr distance)

The 2-way-identification metrics are lifted from ENIGMA (evaluate_recons.py). The two EMD metrics
are JMVR's own; the paper doesn't publish exact formulas, so we use the standard reading (per-channel
1-D Wasserstein on CIELab; 1-D Wasserstein on normalized Depth-Anything-v2 depth). Absolute values may
differ from the paper's scale, but the ranking across our runs is what we use them for.

Inputs are two equal-length lists of PIL RGB images (ground truth, reconstruction).
"""
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from skimage.color import rgb2gray, rgb2lab
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights, Inception_V3_Weights, alexnet, inception_v3,
)
from torchvision.models.feature_extraction import create_feature_extractor

_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_CLIPNORM = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


def _batch(pils, size=224):
    return torch.stack([transforms.functional.to_tensor(p.resize((size, size))) for p in pils])


@torch.no_grad()
def _two_way(recons, images, model, preprocess, feature_layer, device):
    """2-AFC identification success rate (ENIGMA). recons/images: [N,3,H,W] in [0,1]."""
    N = len(images)
    preds = model(torch.stack([preprocess(r) for r in recons]).to(device))
    reals = model(torch.stack([preprocess(i) for i in images]).to(device))
    if feature_layer is not None:
        preds, reals = preds[feature_layer], reals[feature_layer]
    preds = preds.float().flatten(1).cpu().numpy()
    reals = reals.float().flatten(1).cpu().numpy()
    r = np.corrcoef(reals, preds)[:N, N:]  # corr(real_i, pred_j)
    congruents = np.diag(r)
    succ = [(np.sum(r[:, i] < congruents[i]) - 1) / (N - 1) for i in range(N)]
    return float(np.mean(succ))


@torch.no_grad()
def _corr_distance(recons, images, model, preprocess, device):
    """Mean correlation distance in a feature space (SwAV metric, lower better)."""
    g = model(torch.stack([preprocess(i) for i in images]).to(device))["avgpool"].flatten(1).cpu().numpy()
    f = model(torch.stack([preprocess(r) for r in recons]).to(device))["avgpool"].flatten(1).cpu().numpy()
    return float(np.mean([sp.spatial.distance.correlation(g[i], f[i]) for i in range(len(g))]))


def _pix_corr(gts, recs):
    vals = []
    for g, r in zip(gts, recs):
        a = np.asarray(g.resize((425, 425)), np.float32).ravel()
        b = np.asarray(r.resize((425, 425)), np.float32).ravel()
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals))


def _ssim(gts, recs):
    vals = []
    for g, r in zip(gts, recs):
        gg = rgb2gray(np.asarray(g.resize((425, 425))) / 255.0)
        rr = rgb2gray(np.asarray(r.resize((425, 425))) / 255.0)
        vals.append(structural_similarity(gg, rr, gaussian_weights=True, sigma=1.5,
                                          use_sample_covariance=False, data_range=1.0))
    return float(np.mean(vals))


def _lab_emd(gts, recs):
    """CIELab colour EMD (lower = closer colour distribution). Sum of per-channel 1-D Wasserstein."""
    vals = []
    for g, r in zip(gts, recs):
        lg = rgb2lab(np.asarray(g.resize((224, 224))) / 255.0)
        lr = rgb2lab(np.asarray(r.resize((224, 224))) / 255.0)
        vals.append(sum(wasserstein_distance(lg[..., c].ravel(), lr[..., c].ravel()) for c in range(3)))
    return float(np.mean(vals))


def _depth_emd(gts, recs, device, cache_dir):
    """Depth-Anything-v2 depth-map EMD (lower = closer spatial layout). 1-D Wasserstein on norm depth."""
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if device.type == "cuda" else -1)

    def depth(pil):
        d = pipe(pil.resize((224, 224)))["predicted_depth"].squeeze().float().cpu().numpy()
        return ((d - d.min()) / (d.max() - d.min() + 1e-8)).ravel()

    vals = [wasserstein_distance(depth(g), depth(r)) for g, r in zip(gts, recs)]
    del pipe
    torch.cuda.empty_cache()
    return float(np.mean(vals))


def compute_jmvr_metrics(gts, recs, device, cache_dir=None):
    """Return dict of mean JMVR metrics over the (gts, recs) PIL image lists."""
    import open_clip

    gt_t, rec_t = _batch(gts), _batch(recs)
    out = {}
    out["PixCorr"] = _pix_corr(gts, recs)
    out["SSIM"] = _ssim(gts, recs)
    out["LabEMD"] = _lab_emd(gts, recs)

    alex = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                                    return_nodes=["features.4", "features.11"]).to(device).eval()
    pp_alex = transforms.Compose([transforms.Resize(256), _IMAGENET])
    out["AlexNet(2)"] = _two_way(rec_t, gt_t, alex, pp_alex, "features.4", device)
    out["AlexNet(5)"] = _two_way(rec_t, gt_t, alex, pp_alex, "features.11", device)
    del alex; torch.cuda.empty_cache()

    incep = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT),
                                     return_nodes=["avgpool"]).to(device).eval()
    pp_incep = transforms.Compose([transforms.Resize(342), _IMAGENET])
    out["Inception"] = _two_way(rec_t, gt_t, incep, pp_incep, "avgpool", device)
    del incep; torch.cuda.empty_cache()

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai",
                                                             device=device, cache_dir=cache_dir)
    clip_model.eval()
    pp_clip = transforms.Compose([transforms.Resize(224), _CLIPNORM])
    out["CLIP"] = _two_way(rec_t, gt_t, clip_model.encode_image, pp_clip, None, device)

    @torch.no_grad()
    def _clip_cos():
        e = lambda t: nn.functional.normalize(clip_model.encode_image(
            torch.stack([pp_clip(x) for x in t]).to(device)), dim=-1)
        return float((e(rec_t) * e(gt_t)).sum(-1).mean().cpu())
    out["CLIP_cos"] = _clip_cos()
    del clip_model; torch.cuda.empty_cache()

    swav = create_feature_extractor(torch.hub.load("facebookresearch/swav:main", "resnet50"),
                                    return_nodes=["avgpool"]).to(device).eval()
    pp_swav = transforms.Compose([transforms.Resize(224), _IMAGENET])
    out["SwAV"] = _corr_distance(rec_t, gt_t, swav, pp_swav, device)
    del swav; torch.cuda.empty_cache()

    out["DepthEMD"] = _depth_emd(gts, recs, device, cache_dir)
    return out
