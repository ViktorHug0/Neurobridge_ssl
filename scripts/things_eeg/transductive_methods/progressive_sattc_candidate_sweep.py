#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from module.util import (
    apply_orthogonal_map,
    fit_soft_assignment_procrustes,
    score_query_features,
    sinkhorn_normalize,
    subject_adaptive_whiten,
    topk,
)
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything


def _load_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_rows(features, eps=1e-12):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, eps, None)


def _refine_scores(query_features, image_features, eval_mode, sattc_params):
    use_csls = eval_mode in {"csls", "saw_csls"}
    csls_k = sattc_params.get("csls_k", 12)
    tau = sattc_params.get("sinkhorn_tau", 0.05)
    n_iters = sattc_params.get("sinkhorn_iters", 20)

    # Note: when candidates < k, score_query_features handles it
    scores, _ = score_query_features(query_features, image_features, use_csls=use_csls, csls_k=csls_k)

    if sattc_params.get("soft_procrustes_enabled", False):
        for _ in range(max(1, int(sattc_params.get("soft_procrustes_steps", 1)))):
            assignment = sinkhorn_normalize(scores, tau=tau, num_iters=n_iters)
            ortho_weights = fit_soft_assignment_procrustes(
                query_features,
                image_features,
                assignment,
                power=sattc_params.get("soft_procrustes_power", 1.0),
                normalize_inputs=sattc_params.get("soft_procrustes_normalize_inputs", False),
            )
            if ortho_weights is None:
                break
            query_features = apply_orthogonal_map(query_features, ortho_weights)
            scores, _ = score_query_features(query_features, image_features, use_csls=use_csls, csls_k=csls_k)

    if sattc_params.get("sinkhorn_enabled", False):
        scores = sinkhorn_normalize(scores, tau=tau, num_iters=n_iters)

    return scores


def _score_variant(processed_query, candidate_features, target_indices, eval_mode, sattc_params, method_name):
    if method_name == "full_sattc":
        similarity_matrix = _refine_scores(processed_query, candidate_features, eval_mode, sattc_params)
    elif method_name == "saw_only":
        similarity_matrix, _ = score_query_features(processed_query, candidate_features, use_csls=(eval_mode in {"csls", "saw_csls"}), csls_k=sattc_params["csls_k"])
    elif method_name == "plain_cosine":
        similarity_matrix, _ = score_query_features(processed_query, candidate_features, use_csls=False, csls_k=0)
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    top5_count, top1_count = topk(similarity_matrix, 5, target_indices=target_indices)
    return round(100.0 * top1_count / len(target_indices), 4), round(100.0 * top5_count / len(target_indices), 4)


def _process_query_features_safe(query_features, eval_mode, sattc_params):
    query_features = np.asarray(query_features, dtype=np.float32)
    if eval_mode in {"plain_cosine", "csls"}:
        return query_features
    if eval_mode not in {"saw", "saw_csls"}:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    if query_features.shape[0] <= 1:
        if sattc_params.get("saw_renorm", True):
            return _normalize_rows(query_features)
        return query_features.copy()

    subject_ids = np.zeros(query_features.shape[0], dtype=np.int64)
    return subject_adaptive_whiten(
        query_features,
        subject_ids,
        shrink=sattc_params.get("saw_shrink", 0.2),
        diag=sattc_params.get("saw_diag", False),
        normalize=sattc_params.get("saw_renorm", True),
    )


def _find_checkpoint_dir(source_run_dir, subject_id):
    suffix = f"-sub-{int(subject_id):02d}"
    matches = [
        os.path.join(source_run_dir, name)
        for name in os.listdir(source_run_dir)
        if name.endswith(suffix) and os.path.isdir(os.path.join(source_run_dir, name))
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find checkpoint directory for subject {subject_id} in '{source_run_dir}'.")
    return max(matches, key=os.path.getmtime)


def _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id):
    merged = {}
    merged.update(train_cfg)
    merged.update(eval_cfg)
    merged["test_subject_id"] = int(subject_id)
    merged["device"] = runtime_args.device
    merged["eval_batch_size"] = runtime_args.batch_size
    merged["num_workers"] = runtime_args.num_workers
    return SimpleNamespace(**merged)


def _build_test_dataset(eval_args, subject_id):
    average = _to_bool(getattr(eval_args, "data_average", True), True)
    return EEGPreImageDataset(
        subject_ids=[int(subject_id)],
        eeg_data_dir=eval_args.eeg_data_dir,
        selected_channels=eval_args.selected_channels,
        time_window=eval_args.time_window,
        image_feature_dir=eval_args.image_feature_dir,
        text_feature_dir=getattr(eval_args, "text_feature_dir", ""),
        image_aug=False,
        aug_image_feature_dirs=[],
        average=average,
        _random=False,
        eeg_transform=None,
        train=False,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=_to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )


def _load_modules(eval_args, checkpoint_dir, test_dataset):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_test_best.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at '{checkpoint_path}'.")

    device = torch.device(eval_args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    eeg_sample_points = test_dataset.num_sample_points
    channels_num = test_dataset.channels_num
    image_feature_dim = test_dataset.image_features.shape[-1]
    backbone_feature_dim = getattr(eval_args, "eeg_backbone_dim", 0) or image_feature_dim

    model = build_eeg_encoder(eval_args, backbone_feature_dim, eeg_sample_points, channels_num).to(device)
    img_projector = build_projector(eval_args.projector, image_feature_dim, eval_args.feature_dim).to(device)
    architecture = getattr(eval_args, "architecture", checkpoint.get("architecture", "baseline"))
    if architecture != "baseline":
        raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")
    eeg_projector = build_projector(eval_args.projector, backbone_feature_dim, eval_args.feature_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])

    model.eval()
    eeg_projector.eval()
    img_projector.eval()

    return {
        "model": model,
        "eeg_projector": eeg_projector,
        "img_projector": img_projector,
        "architecture": architecture,
        "device": device,
    }


def _forward_feature(modules, eeg_backbone_batch):
    return modules["eeg_projector"](eeg_backbone_batch)


def _encode_subject_features(eval_args, modules, dataset):
    loader = DataLoader(
        dataset,
        batch_size=getattr(eval_args, "eval_batch_size", 200),
        shuffle=False,
        num_workers=getattr(eval_args, "num_workers", 0),
    )

    eeg_feature_list = []
    image_feature_list = []
    object_idx_list = []
    image_idx_list = []

    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(modules["device"])
            image_feature_batch = batch[1].to(modules["device"])
            subject_id_batch = batch[3].to(modules["device"])
            object_idx_batch = batch[4]
            image_idx_batch = batch[5]

            eeg_backbone_batch = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_id_batch)
            eeg_feature_batch = _forward_feature(modules, eeg_backbone_batch)
            image_feature_proj = modules["img_projector"](image_feature_batch)

            eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
            image_feature_list.append(image_feature_proj.cpu().numpy())
            object_idx_list.append(object_idx_batch.numpy())
            image_idx_list.append(image_idx_batch.numpy())

    return {
        "query_features": np.concatenate(eeg_feature_list, axis=0),
        "image_features": np.concatenate(image_feature_list, axis=0),
        "object_indices": np.concatenate(object_idx_list, axis=0),
        "image_indices": np.concatenate(image_idx_list, axis=0),
    }


def _selection_seed(base_seed, subject_id, sample_count):
    return int(base_seed) * 100000 + int(subject_id) * 1000 + int(sample_count)


def _resolve_sample_count_range(total_candidates, runtime_args):
    if runtime_args.sample_counts:
        return [c for c in runtime_args.sample_counts if c <= total_candidates]
    min_samples = max(1, int(runtime_args.min_samples))
    max_samples = total_candidates if runtime_args.max_samples is None else int(runtime_args.max_samples)
    max_samples = min(max_samples, total_candidates)
    return range(min_samples, max_samples + 1, runtime_args.sample_step)


def _evaluate_progressive_subject(subject_id, checkpoint_dir, runtime_args):
    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
    test_dataset = _build_test_dataset(eval_args, subject_id)
    
    modules = _load_modules(eval_args, checkpoint_dir, test_dataset)
    encoded = _encode_subject_features(eval_args, modules, test_dataset)
    query_features_all = encoded["query_features"]
    image_features_all = encoded["image_features"]
    total_candidates = image_features_all.shape[0]

    # Configs for the 3 methods
    best_sattc_params = {
        "saw_shrink": runtime_args.sattc_saw_shrink,
        "saw_diag": runtime_args.sattc_saw_diag,
        "saw_renorm": not runtime_args.sattc_saw_no_renorm,
        "csls_k": runtime_args.sattc_csls_k,
        "sinkhorn_enabled": True,
        "sinkhorn_tau": runtime_args.sattc_sinkhorn_tau,
        "sinkhorn_iters": runtime_args.sattc_sinkhorn_iters,
        "soft_procrustes_enabled": True,
        "soft_procrustes_steps": runtime_args.sattc_soft_procrustes_steps,
        "soft_procrustes_power": runtime_args.sattc_soft_procrustes_power,
        "soft_procrustes_normalize_inputs": runtime_args.sattc_soft_procrustes_normalize_inputs,
    }
    saw_csls_k5_params = {
        "saw_shrink": runtime_args.sattc_saw_shrink,
        "saw_diag": runtime_args.sattc_saw_diag,
        "saw_renorm": not runtime_args.sattc_saw_no_renorm,
        "csls_k": 5,
        "sinkhorn_enabled": False,
        "soft_procrustes_enabled": False,
    }

    sample_count_range = _resolve_sample_count_range(total_candidates, runtime_args)
    rows = []
    for base_seed in runtime_args.seeds:
        for sample_count in sample_count_range:
            selection_seed = _selection_seed(base_seed, subject_id, sample_count)
            rng = np.random.default_rng(selection_seed)
            selected_indices = rng.choice(total_candidates, size=sample_count, replace=False)
            
            query_subset = query_features_all[selected_indices]
            
            # --- TWO REGIMES ---
            for regime_type in ["matching", "all"]:
                if regime_type == "matching":
                    # EEG subset vs exactly the matching image subset
                    candidates = image_features_all[selected_indices]
                    targets = np.arange(sample_count)
                else:
                    # EEG subset vs all 200 candidates
                    candidates = image_features_all
                    targets = selected_indices

                # 1. Plain Cosine
                top1_plain, top5_plain = _score_variant(query_subset, candidates, targets, "plain_cosine", {"csls_k": 0}, "plain_cosine")
                rows.append({
                    "row_type": "subject", "regime": regime_type, "method": "plain_cosine", "seed": int(base_seed),
                    "subject_id": int(subject_id), "sample_count": int(sample_count),
                    "top1_acc": top1_plain, "top5_acc": top5_plain,
                })

                # 2. Whitening (SAW + 5-k CSLS)
                proc_saw = _process_query_features_safe(query_subset, "saw_csls", saw_csls_k5_params)
                top1_saw, top5_saw = _score_variant(proc_saw, candidates, targets, "saw_csls", saw_csls_k5_params, "saw_only")
                rows.append({
                    "row_type": "subject", "regime": regime_type, "method": "saw_csls_k5", "seed": int(base_seed),
                    "subject_id": int(subject_id), "sample_count": int(sample_count),
                    "top1_acc": top1_saw, "top5_acc": top5_saw,
                })

                # 3. Geometric Alignment (Full TTA)
                proc_full = _process_query_features_safe(query_subset, "saw_csls", best_sattc_params)
                top1_full, top5_full = _score_variant(proc_full, candidates, targets, "saw_csls", best_sattc_params, "full_sattc")
                rows.append({
                    "row_type": "subject", "regime": regime_type, "method": "full_sattc", "seed": int(base_seed),
                    "subject_id": int(subject_id), "sample_count": int(sample_count),
                    "top1_acc": top1_full, "top5_acc": top5_full,
                })

    return rows


def _build_summary_rows(subject_df):
    # 1. Average across subjects for each (regime, method, seed, sample_count)
    # This gives us 3 observations (one per seed) for each experimental point.
    seed_level_df = (
        subject_df.groupby(["regime", "method", "seed", "sample_count"], as_index=False)[["top1_acc", "top5_acc"]]
        .mean()
    )

    # 2. Compute mean and std across the 3 seeds
    grand_average = (
        seed_level_df.groupby(["regime", "method", "sample_count"], as_index=False).agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top5_mean=("top5_acc", "mean"),
            top5_std=("top5_acc", "std"),
        )
        .sort_values(["regime", "method", "sample_count"])
    )
    
    # Handle cases where std is NaN (e.g. only 1 seed)
    grand_average["top1_std"] = grand_average["top1_std"].fillna(0.0)
    grand_average["top5_std"] = grand_average["top5_std"].fillna(0.0)
    return grand_average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_dir", type=str, required=True)
    parser.add_argument("--held_out_subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expected_test_size", type=int, default=200)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--sample_step", type=int, default=5)
    parser.add_argument("--sample_counts", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2099, 2100, 2101])
    parser.add_argument("--eval_mode", type=str, default="saw_csls")
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.7)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_csls_k", type=int, default=1)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.08)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=10)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=6)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.0)
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seeds[0])

    all_subject_rows = []
    for subject_id in args.held_out_subjects:
        checkpoint_dir = _find_checkpoint_dir(args.source_run_dir, subject_id)
        print(f"[progressive_sattc] subject={subject_id:02d}")
        subject_rows = _evaluate_progressive_subject(subject_id, checkpoint_dir, args)
        all_subject_rows.extend(subject_rows)

    subject_df = pd.DataFrame(all_subject_rows)
    summary_df = _build_summary_rows(subject_df)
    summary_csv = os.path.join(args.output_dir, "progressive_sattc_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved results to: {summary_csv}")


if __name__ == "__main__":
    main()
