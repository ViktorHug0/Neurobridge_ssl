#!/usr/bin/env python3
"""Measure alignment active-feature fraction on the full train split at checkpoint_test_best."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from train import (
    _empty_alignment_sparsity_accumulator,
    _finalize_alignment_sparsity_accumulator,
    _update_alignment_sparsity_accumulator,
    build_eeg_encoder,
    build_projector,
    run_eeg_backbone,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    repo = REPO_ROOT
    default_session = os.path.join(
        repo,
        "results/things_eeg/inter-subject-sparse/"
        "sparse_clip_confidence_bb64_seed3300_20260527-172056",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session_dir", default=default_session)
    p.add_argument(
        "--cache_csv",
        default=None,
        help="Per-run cache CSV (default: <session_dir>/train_sparsity_at_best.csv)",
    )
    p.add_argument(
        "--merge_sweep",
        default=None,
        help="Write sweep CSV with train columns (default: <session_dir>/sweep_summary.csv)",
    )
    p.add_argument("--device", default=None, help="Override device (default: from each train_config)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--skip_existing", action="store_true", help="Skip runs already in cache")
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def collect_train_sparsity(cfg: dict, checkpoint_path: str, device: torch.device, batch_size: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_subject_ids = [int(x) for x in cfg["train_subject_ids"]]

    dataset = EEGPreImageDataset(
        train_subject_ids,
        cfg["eeg_data_dir"],
        cfg.get("selected_channels", []),
        cfg["time_window"],
        cfg["image_feature_dir"],
        cfg.get("text_feature_dir", ""),
        bool(cfg.get("image_aug", False)),
        cfg.get("aug_image_feature_dirs", []),
        bool(cfg.get("data_average", True)),
        bool(cfg.get("data_random", False)),
        None,
        True,
        bool(cfg.get("image_test_aug", False)),
        bool(cfg.get("eeg_test_aug", False)),
        bool(cfg.get("frozen_eeg_prior", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

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
    eeg_acc = _empty_alignment_sparsity_accumulator()
    img_acc = _empty_alignment_sparsity_accumulator()
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            subject_id_batch = batch[3].to(device)
            eeg_backbone = run_eeg_backbone(model, args_ns, eeg_batch, subject_id_batch)
            eeg_feat = eeg_projector(eeg_backbone)
            img_feat = img_projector(image_feature_batch)
            _update_alignment_sparsity_accumulator(eeg_acc, eeg_feat.cpu().numpy())
            _update_alignment_sparsity_accumulator(img_acc, img_feat.cpu().numpy())

    eeg_sp = _finalize_alignment_sparsity_accumulator(eeg_acc)
    img_sp = _finalize_alignment_sparsity_accumulator(img_acc)
    return {
        "n_train_samples": len(dataset),
        "eeg_active_feat_frac_train": eeg_sp["active_feat_frac"],
        "eeg_l0_frac_train": eeg_sp["l0_frac"],
        "eeg_l0_mean_train": eeg_sp["l0_mean"],
        "img_active_feat_frac_train": img_sp["active_feat_frac"],
        "img_l0_frac_train": img_sp["l0_frac"],
        "best_epoch": checkpoint.get("epoch"),
    }


def discover_runs(session_dir: str) -> list[tuple[str, str, str]]:
    """Return (config_name, sub_label, run_dir) for runs with best checkpoint."""
    runs = []
    for config_dir in sorted(glob.glob(os.path.join(session_dir, "sparse_fd*"))):
        if not os.path.isdir(config_dir):
            continue
        config_name = os.path.basename(config_dir)
        for run_dir in sorted(glob.glob(os.path.join(config_dir, "*-sub-*"))):
            ckpt = os.path.join(run_dir, "checkpoint_test_best.pth")
            cfg_path = os.path.join(run_dir, "train_config.json")
            if os.path.isfile(ckpt) and os.path.isfile(cfg_path):
                sub_label = run_dir.rsplit("-", 1)[-1]
                runs.append((config_name, sub_label, run_dir))
    return runs


def merge_into_sweep(session_dir: str, cache_df: pd.DataFrame, sweep_path: str) -> None:
    if not os.path.isfile(sweep_path):
        raise FileNotFoundError(sweep_path)
    sweep = pd.read_csv(sweep_path)
    sweep.columns = [c.strip() for c in sweep.columns]

    per_run = cache_df.copy()
    per_run["sub"] = per_run["sub_label"]
    agg = (
        per_run.groupby("config_name", as_index=False)[
            [
                "eeg_active_feat_frac_train",
                "img_active_feat_frac_train",
                "eeg_l0_frac_train",
            ]
        ]
        .mean()
    )
    agg = agg.rename(columns={"config_name": "config"})
    sweep = sweep.drop(
        columns=[
            c
            for c in (
                "eeg_active_feat_frac_train",
                "img_active_feat_frac_train",
                "eeg_l0_frac_train",
            )
            if c in sweep.columns
        ],
        errors="ignore",
    )
    sweep = sweep.merge(agg, on="config", how="left")
    sweep.to_csv(sweep_path, index=False)
    print(f"Updated sweep with train columns: {sweep_path}")


def main() -> None:
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    cache_csv = args.cache_csv or os.path.join(session_dir, "train_sparsity_at_best.csv")
    merge_sweep = args.merge_sweep
    if merge_sweep is None:
        merge_sweep = os.path.join(session_dir, "sweep_summary.csv")

    existing = {}
    if os.path.isfile(cache_csv):
        prev = pd.read_csv(cache_csv)
        for _, row in prev.iterrows():
            existing[row["run_dir"]] = row.to_dict()

    rows = list(existing.values())
    runs = discover_runs(session_dir)
    print(f"Found {len(runs)} runs with checkpoint_test_best.pth under {session_dir}", flush=True)

    for config_name, sub_label, run_dir in runs:
        if args.skip_existing and run_dir in existing:
            continue
        cfg = _load_json(os.path.join(run_dir, "train_config.json"))
        device_str = args.device or cfg.get("device", "cuda:0")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)
        seed_everything(cfg.get("seed", 0))
        ckpt_path = os.path.join(run_dir, "checkpoint_test_best.pth")
        print(f"Measuring train sparsity: {config_name} / {sub_label} ...", flush=True)
        sp = collect_train_sparsity(cfg, ckpt_path, device, args.batch_size)
        row = {
            "config_name": config_name,
            "sub_label": sub_label,
            "run_dir": run_dir,
            **sp,
        }
        rows.append(row)
        existing[run_dir] = row
        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        print(
            f"  eeg_active_train={sp['eeg_active_feat_frac_train']:.4f} "
            f"(n={sp['n_train_samples']}, best_epoch={sp['best_epoch']})",
            flush=True,
        )

    out = pd.DataFrame(rows)
    print(f"Saved cache ({len(out)} runs): {cache_csv}", flush=True)

    if merge_sweep:
        merge_into_sweep(session_dir, out, os.path.abspath(merge_sweep))


if __name__ == "__main__":
    main()
