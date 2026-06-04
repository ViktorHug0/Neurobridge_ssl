#!/usr/bin/env python3
"""Analyze sparsity and object-coherence of alignment embeddings (Sparse CLIP-style)."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from train import (
    build_eeg_encoder,
    build_projector,
    compute_alignment_sparsity_stats,
    run_eeg_backbone,
    seed_everything,
)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _encode_test_features(cfg, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    test_subject_id = int(cfg["test_subject_ids"][0])

    dataset = EEGPreImageDataset(
        [test_subject_id],
        cfg["eeg_data_dir"],
        cfg.get("selected_channels", []),
        cfg["time_window"],
        cfg["image_feature_dir"],
        cfg.get("text_feature_dir", ""),
        False,
        cfg.get("aug_image_feature_dirs", []),
        bool(cfg.get("data_average", True)),
        False,
        None,
        False,
        bool(cfg.get("image_test_aug", False)),
        bool(cfg.get("eeg_test_aug", False)),
        bool(cfg.get("frozen_eeg_prior", False)),
    )
    loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=0)

    image_feature_dim = dataset.image_features.shape[-1]
    backbone_feature_dim = cfg.get("eeg_backbone_dim", 0) or image_feature_dim
    projector_activation = cfg.get("projector_activation", "none")
    projector_topk = cfg.get("projector_topk", 512)

    model = build_eeg_encoder(
        SimpleNamespace(**cfg),
        backbone_feature_dim,
        dataset.num_sample_points,
        dataset.channels_num,
    ).to(device)
    eeg_projector = build_projector(
        cfg["projector"],
        backbone_feature_dim,
        cfg["feature_dim"],
        activation=projector_activation,
        topk=projector_topk,
    ).to(device)
    img_projector = build_projector(
        cfg["projector"],
        image_feature_dim,
        cfg["feature_dim"],
        activation=projector_activation,
        topk=projector_topk,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    model.eval()
    eeg_projector.eval()
    img_projector.eval()

    args_ns = SimpleNamespace(**cfg)
    eeg_list, img_list, obj_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].cpu().numpy()

            eeg_backbone = run_eeg_backbone(model, args_ns, eeg_batch, subject_id_batch)
            eeg_feat = eeg_projector(eeg_backbone).cpu().numpy()
            img_feat = img_projector(image_feature_batch).cpu().numpy()
            eeg_list.append(eeg_feat)
            img_list.append(img_feat)
            obj_list.append(object_idx_batch)

    return (
        np.concatenate(eeg_list, axis=0),
        np.concatenate(img_list, axis=0),
        np.concatenate(obj_list, axis=0),
    )


def object_coherence_per_dim(features, tau=0.001, n_min=2):
    """EEG-adapted Clarity (Sparse CLIP Eq. 2): mean pairwise cosine among trials activating each dim."""
    features = np.asarray(features, dtype=np.float32)
    n_samples, n_dims = features.shape
    dim_scores = []
    active_dim_count = 0

    for dim_idx in range(n_dims):
        activations = features[:, dim_idx]
        mask = activations > tau
        n_active = int(mask.sum())
        if n_active < n_min:
            continue
        active_dim_count += 1
        feats = features[mask]
        if n_active < 2:
            continue
        sim_mat = cosine_similarity(feats)
        triu = sim_mat[np.triu_indices(n_active, k=1)]
        if triu.size > 0:
            dim_scores.append(float(triu.mean()))

    mean_coherence = float(np.mean(dim_scores)) if dim_scores else 0.0
    return {
        "mean_object_coherence": mean_coherence,
        "n_scored_dims": len(dim_scores),
        "n_active_dims": active_dim_count,
    }


def top_objects_per_dim(features, object_ids, top_k=5, tau=0.001):
    rows = []
    for dim_idx in range(features.shape[1]):
        vals = features[:, dim_idx]
        mask = vals > tau
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        order = idx[np.argsort(-vals[idx])][:top_k]
        for rank, sample_i in enumerate(order):
            rows.append({
                "dim": dim_idx,
                "rank": rank + 1,
                "object_idx": int(object_ids[sample_i]),
                "activation": float(vals[sample_i]),
            })
    return pd.DataFrame(rows)


def analyze_run_dir(run_dir, checkpoint_name="checkpoint_test_best.pth", tau=0.001, n_min=2, top_k=5):
    cfg_path = os.path.join(run_dir, "train_config.json")
    ckpt_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.isfile(cfg_path) or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Need train_config.json and {checkpoint_name} in {run_dir}")

    cfg = _load_json(cfg_path)
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get("seed"))

    eeg_feat, img_feat, obj_ids = _encode_test_features(cfg, ckpt_path, device)
    eeg_sp = compute_alignment_sparsity_stats(eeg_feat)
    img_sp = compute_alignment_sparsity_stats(img_feat)
    eeg_coh = object_coherence_per_dim(eeg_feat, tau=tau, n_min=n_min)

    top_df = top_objects_per_dim(eeg_feat, obj_ids, top_k=top_k, tau=tau)
    out_prefix = run_dir
    top_path = os.path.join(out_prefix, "sparse_top_objects_eeg.csv")
    top_df.to_csv(top_path, index=False)

    summary = {
        "run_dir": run_dir,
        **{f"eeg_{k}": v for k, v in eeg_sp.items()},
        **{f"img_{k}": v for k, v in img_sp.items()},
        **{f"eeg_{k}": v for k, v in eeg_coh.items()},
        "top_objects_csv": top_path,
    }
    summary_path = os.path.join(out_prefix, "sparse_alignment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Training run folder with train_config.json and checkpoint")
    parser.add_argument(
        "--baseline_run_dir",
        default=None,
        help="Optional dense baseline run dir for the same held-out subject",
    )
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary = analyze_run_dir(run_dir, tau=args.tau, n_min=args.n_min, top_k=args.top_k)
    print(json.dumps(summary, indent=2))

    if args.baseline_run_dir:
        baseline_dir = os.path.abspath(args.baseline_run_dir)
        baseline_summary = analyze_run_dir(baseline_dir, tau=args.tau, n_min=args.n_min, top_k=args.top_k)
        compare = {
            "sparse_eeg_l0_frac": summary["eeg_l0_frac"],
            "baseline_eeg_l0_frac": baseline_summary["eeg_l0_frac"],
            "sparse_eeg_mean_object_coherence": summary["eeg_mean_object_coherence"],
            "baseline_eeg_mean_object_coherence": baseline_summary["eeg_mean_object_coherence"],
        }
        compare_path = os.path.join(run_dir, "sparse_vs_baseline.json")
        with open(compare_path, "w") as f:
            json.dump(compare, f, indent=2)
        print("Comparison:", json.dumps(compare, indent=2))


if __name__ == "__main__":
    main()
