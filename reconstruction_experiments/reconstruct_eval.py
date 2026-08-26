"""EEG -> image reconstruction eval (within-subject add-on).

Loads a trained checkpoint (aligned to raw ViT-H/14), predicts a CLIP image embedding for
each of the 200 test images, reconstructs them with SDXL-Turbo + IP-Adapter, and scores
recon-vs-ground-truth with the full JMVR metric suite (module/recon_metrics.py).

Reuses the encoder/projector build + feature encoding from evaluate.py; the only new code is
the reconstruct loop + metrics. --lowlevel_decoder switches the generator to img2img off an
EEG-decoded VAE latent (trades semantics for pixel fidelity; see train_lowlevel.py).
"""
import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from module.dataset import EEGPreImageDataset
from train import build_eeg_encoder, build_projector
from evaluate import _encode_dataset_features, _to_bool
from torch.utils.data import DataLoader

from module.reconstruct import SDXLReconstructor
from module.recon_metrics import compute_jmvr_metrics
from module.lowlevel import EEGLatentDecoder


def load_and_encode(checkpoint_dir, device, rescale_norm=0.0):
    """Return (emb_by_object [200,1024], order preserved by object index).

    rescale_norm>0 forces every prediction to that L2 norm (use for models trained fully
    L2-normalized). Default 0 keeps the raw predicted magnitude -- required for ENIGMA-faithful
    (mse_on_raw) models, which learn the true CLIP magnitude, and for the confidence runs, which
    encode per-trial confidence in that magnitude.
    """
    train_cfg = json.load(open(os.path.join(checkpoint_dir, "train_config.json")))
    a = SimpleNamespace(**train_cfg)
    a.device = str(device)
    a.test_subject_id = a.test_subject_ids[0]
    a.eval_batch_size = 200
    a.num_workers = getattr(a, "num_workers", 0)

    ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint_test_best.pth"), map_location=device)

    ds = EEGPreImageDataset(
        [a.test_subject_id], a.eeg_data_dir, a.selected_channels, a.time_window,
        a.image_feature_dir, getattr(a, "text_feature_dir", ""), False,
        getattr(a, "aug_image_feature_dirs", []), True, False, None, False,
        _to_bool(getattr(a, "image_test_aug", False)),
        _to_bool(getattr(a, "eeg_test_aug", False)),
        _to_bool(getattr(a, "frozen_eeg_prior", False)),
    )
    img_dim = ds.image_features.shape[-1]
    backbone_dim = getattr(a, "eeg_backbone_dim", 0) or img_dim
    act = getattr(a, "projector_activation", "none")
    topk = getattr(a, "projector_topk", 512)

    model = build_eeg_encoder(a, backbone_dim, ds.num_sample_points, ds.channels_num).to(device)
    eeg_proj = build_projector(a.projector, backbone_dim, a.feature_dim, activation=act, topk=topk).to(device)
    img_proj = build_projector(a.projector, img_dim, a.feature_dim, activation=act, topk=topk).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_proj.load_state_dict(ckpt["eeg_projector_state_dict"])
    img_proj.load_state_dict(ckpt["img_projector_state_dict"])
    model.eval(); eeg_proj.eval(); img_proj.eval()

    eeg_all, _, _, object_all, _, _ = _encode_dataset_features(
        a, {"eeg_projector": eeg_proj, "img_projector": img_proj},
        model, img_proj, device, [a.test_subject_id], average=True,
    )
    # reorder rows so index == object (test set: one image per object)
    order = np.argsort(object_all)
    eeg_all = eeg_all[order].astype(np.float32)
    if rescale_norm and rescale_norm > 0:
        eeg_all = eeg_all / np.clip(np.linalg.norm(eeg_all, axis=1, keepdims=True), 1e-8, None) * rescale_norm
    return eeg_all


def encode_lowlevel_latents(checkpoint_dir, decoder_path, device):
    """Run the EEG->VAE-latent decoder on the test set -> (200,4,64,64) ordered by object."""
    cfg = json.load(open(os.path.join(checkpoint_dir, "train_config.json")))
    ds = EEGPreImageDataset(
        [cfg["test_subject_ids"][0]], cfg["eeg_data_dir"], cfg["selected_channels"],
        cfg["time_window"], cfg["image_feature_dir"], "", False, [], average=True, train=False)
    ckpt = torch.load(decoder_path, map_location=device)
    model = EEGLatentDecoder(ckpt["channels"], ckpt["T"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    lats, objs = [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=200, shuffle=False):
            lats.append(model(batch[0].to(device).float()).cpu().numpy())
            objs.append(batch[4].numpy())
    lats, objs = np.concatenate(lats), np.concatenate(objs)
    return lats[np.argsort(objs)]


def load_gt_images(image_set_dir, n):
    dirs = sorted(glob.glob(os.path.join(image_set_dir, "test_images", "*/")))[:n]
    imgs = []
    for d in dirs:
        p = sorted(glob.glob(os.path.join(d, "*")))[0]
        imgs.append(Image.open(p).convert("RGB").resize((224, 224)))
    return imgs


def contact_sheet(gts, recs, path, cols=10):
    n = len(gts); rows = (n + cols - 1) // cols
    W = H = 112
    sheet = Image.new("RGB", (cols * W, rows * 2 * H), "white")
    for i, (g, r) in enumerate(zip(gts, recs)):
        c, rr = i % cols, i // cols
        sheet.paste(g.resize((W, H)), (c * W, rr * 2 * H))
        sheet.paste(r.resize((W, H)), (c * W, rr * 2 * H + H))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_dir", default="./output/recon")
    ap.add_argument("--image_set_dir", default="/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--rescale_norm", type=float, default=0.0,
                    help="0=use raw predicted magnitude (ENIGMA/confidence models); >0 forces this L2 norm (fully-normalized models, e.g. 22)")
    ap.add_argument("--lowlevel_decoder", default=None, help="EEG->VAE-latent decoder.pth; enables img2img low-level init")
    ap.add_argument("--strength", type=float, default=0.85, help="img2img strength for the low-level init")
    ap.add_argument("--hf_cache", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    emb = load_and_encode(args.checkpoint_dir, device, rescale_norm=args.rescale_norm)[: args.n_images]
    gts = load_gt_images(args.image_set_dir, args.n_images)
    assert len(gts) == len(emb), f"{len(gts)} gt vs {len(emb)} embeddings"

    latents = None
    if args.lowlevel_decoder:
        latents = encode_lowlevel_latents(args.checkpoint_dir, args.lowlevel_decoder, device)[: args.n_images]

    recon = SDXLReconstructor(cache_dir=args.hf_cache, device=str(device))
    recs = []
    for i, e in enumerate(emb):
        init = recon.decode_latent(latents[i]) if latents is not None else None
        img = recon.reconstruct(e, init_image=init, strength=args.strength)
        img.save(os.path.join(args.output_dir, f"recon_{i:03d}.png"))
        recs.append(img)
        if i % 20 == 0:
            print(f"  reconstructed {i+1}/{len(emb)}")

    del recon
    torch.cuda.empty_cache()
    contact_sheet(gts, recs, os.path.join(args.output_dir, "contact_sheet.png"))

    metrics = compute_jmvr_metrics(gts, recs, device, cache_dir=args.hf_cache)
    metrics = {"n": len(recs), **metrics}
    json.dump(metrics, open(os.path.join(args.output_dir, "metrics.json"), "w"), indent=2)
    up = {"PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "Inception", "CLIP", "CLIP_cos"}
    print("\n=== JMVR reconstruction metrics (n=%d) ===" % metrics["n"])
    for k, v in metrics.items():
        if k == "n":
            continue
        arrow = "↑" if k in up else "↓"
        print(f"  {k:12} {v:.4f} {arrow}")
    print(f"  images + contact_sheet.png -> {args.output_dir}")


if __name__ == "__main__":
    main()
