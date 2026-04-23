#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize, topk
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


def _cosine_scores(query_features, image_features):
    query_features = _normalize_rows(query_features)
    image_features = _normalize_rows(image_features)
    return (query_features @ image_features.T).astype(np.float32, copy=False)


def _csls_scores(similarities, k=10):
    similarities = np.asarray(similarities, dtype=np.float32)
    n_q, n_c = similarities.shape
    if n_q == 0 or n_c == 0:
        return similarities
    k_eff = max(1, min(int(k), n_q, n_c))
    rx = np.partition(similarities, kth=n_c - k_eff, axis=1)[:, -k_eff:].mean(axis=1, keepdims=True)
    ry = np.partition(similarities, kth=n_q - k_eff, axis=0)[-k_eff:, :].mean(axis=0, keepdims=True)
    return (2.0 * similarities - rx - ry).astype(np.float32, copy=False)


def _score_query_features(query_features, image_features, use_csls=False, csls_k=12):
    scores = _cosine_scores(query_features, image_features)
    if use_csls:
        scores = _csls_scores(scores, k=csls_k)
    return scores


def _estimate_mu_cov(features, shrink=0.2, diag=False, eps=1e-6):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("Cannot estimate covariance from an empty feature matrix.")
    mu = features.mean(axis=0, keepdims=True)
    centered = features - mu
    cov = (centered.T @ centered / float(max(features.shape[0] - 1, 1))).astype(np.float32, copy=False)
    if diag:
        cov = np.diag(np.diag(cov))
    dim = cov.shape[0]
    trace_mean = float(np.trace(cov)) / max(dim, 1)
    cov = (1.0 - float(shrink)) * cov + float(shrink) * trace_mean * np.eye(dim, dtype=np.float32)
    cov = cov + eps * np.eye(dim, dtype=np.float32)
    return mu.astype(np.float32, copy=False), cov


def _inv_sqrt_cov(cov, eps=1e-6):
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    inv_sqrt = evecs @ np.diag(np.power(evals, -0.5, dtype=np.float32)) @ evecs.T
    return inv_sqrt.astype(np.float32, copy=False)


def _fit_saw_transform(features, shrink=0.2, diag=False, normalize=True):
    mu, cov = _estimate_mu_cov(features, shrink=shrink, diag=diag)
    whitener = _inv_sqrt_cov(cov)
    transformed = (np.asarray(features, dtype=np.float32) - mu) @ whitener
    if normalize:
        transformed = _normalize_rows(transformed)
    return transformed.astype(np.float32, copy=False), {
        "mu": mu.astype(np.float32, copy=False),
        "whitener": whitener.astype(np.float32, copy=False),
        "normalize": bool(normalize),
    }


def _apply_saw_transform(features, saw_stats):
    transformed = (np.asarray(features, dtype=np.float32) - saw_stats["mu"]) @ saw_stats["whitener"]
    if saw_stats["normalize"]:
        transformed = _normalize_rows(transformed)
    return transformed.astype(np.float32, copy=False)


def _project_to_orthogonal(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u @ vt).astype(np.float32, copy=False)


def _blend_orthogonal_map(orthogonal_map, alpha):
    orthogonal_map = np.asarray(orthogonal_map, dtype=np.float32)
    dim = orthogonal_map.shape[0]
    alpha = float(alpha)
    blended = (1.0 - alpha) * np.eye(dim, dtype=np.float32) + alpha * orthogonal_map
    return _project_to_orthogonal(blended)


def _fit_frozen_calibration(query_features, image_features, params):
    transformed, saw_stats = _fit_saw_transform(
        query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
    cumulative_map = np.eye(transformed.shape[1], dtype=np.float32)
    scores = _score_query_features(
        transformed,
        image_features,
        use_csls=params.use_csls,
        csls_k=params.csls_k,
    )
    if params.soft_procrustes_enabled:
        for _ in range(max(1, int(params.soft_procrustes_steps))):
            assignment = sinkhorn_normalize(scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters)
            step_map = fit_soft_assignment_procrustes(
                transformed,
                image_features,
                assignment,
                power=params.soft_procrustes_power,
                normalize_inputs=params.soft_procrustes_normalize_inputs,
            )
            if step_map is None:
                break
            transformed = apply_orthogonal_map(transformed, step_map)
            cumulative_map = (cumulative_map @ np.asarray(step_map, dtype=np.float32)).astype(np.float32, copy=False)
            scores = _score_query_features(
                transformed,
                image_features,
                use_csls=params.use_csls,
                csls_k=params.csls_k,
            )
    return {
        "saw_stats": saw_stats,
        "orthogonal_map": cumulative_map,
    }


def _apply_frozen_calibration(query_features, calibration):
    transformed = _apply_saw_transform(query_features, calibration["saw_stats"])
    transformed = apply_orthogonal_map(transformed, calibration["orthogonal_map"])
    return np.asarray(transformed, dtype=np.float32)


def _evaluate_plain(query_features, image_features):
    scores = _cosine_scores(query_features, image_features)
    top5_count, top1_count = topk(scores, 5)
    total = len(query_features)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_saw_only(query_features, image_features, params):
    transformed, _ = _fit_saw_transform(
        query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
    scores = _cosine_scores(transformed, image_features)
    top5_count, top1_count = topk(scores, 5)
    total = len(transformed)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_saw_transfer(calibration_query_features, query_features, image_features, params):
    _, saw_stats = _fit_saw_transform(
        calibration_query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
    transformed = _apply_saw_transform(query_features, saw_stats)
    scores = _cosine_scores(transformed, image_features)
    top5_count, top1_count = topk(scores, 5)
    total = len(transformed)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_frozen_transfer_with_alpha(query_features, image_features, base_calibration, alpha):
    blended_calibration = {
        "saw_stats": base_calibration["saw_stats"],
        "orthogonal_map": _blend_orthogonal_map(base_calibration["orthogonal_map"], alpha),
    }
    transformed = _apply_frozen_calibration(query_features, blended_calibration)
    scores = _cosine_scores(transformed, image_features)
    top5_count, top1_count = topk(scores, 5)
    total = len(transformed)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id):
    merged = {}
    merged.update(train_cfg)
    merged.update(eval_cfg)
    merged["test_subject_id"] = int(subject_id)
    if runtime_args.device is not None:
        merged["device"] = runtime_args.device
    if runtime_args.batch_size is not None:
        merged["eval_batch_size"] = runtime_args.batch_size
    if runtime_args.num_workers is not None:
        merged["num_workers"] = runtime_args.num_workers
    return SimpleNamespace(**merged)


def _build_test_dataset(eval_args, subject_id):
    average = _to_bool(getattr(eval_args, "data_average", True), True)
    common_kwargs = dict(
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
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=_to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )
    return EEGPreImageDataset(train=False, **common_kwargs)


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


def _encode_subset_features(eval_args, modules, dataset, subset_indices):
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=getattr(eval_args, "eval_batch_size", 200),
        shuffle=False,
        num_workers=getattr(eval_args, "num_workers", 0),
    )

    eeg_feature_list = []
    image_feature_list = []

    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(modules["device"])
            image_feature_batch = batch[1].to(modules["device"])
            subject_id_batch = batch[3].to(modules["device"])

            eeg_backbone_batch = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_id_batch)
            eeg_feature_batch = _forward_feature(modules, eeg_backbone_batch)
            image_feature_proj = modules["img_projector"](image_feature_batch)

            eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
            image_feature_list.append(image_feature_proj.cpu().numpy())

    return (
        np.concatenate(eeg_feature_list, axis=0),
        np.concatenate(image_feature_list, axis=0),
    )


def _format_float_tag(value):
    text = f"{float(value):g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def _resolve_subject_checkpoint_dirs(config_dir):
    pattern = re.compile(r"sub-(\d+)$")
    latest_by_subject = {}
    for name in os.listdir(config_dir):
        path = os.path.join(config_dir, name)
        if not os.path.isdir(path):
            continue
        match = pattern.search(name)
        if match is None:
            continue
        subject_id = int(match.group(1))
        previous = latest_by_subject.get(subject_id)
        if previous is None or os.path.getmtime(path) > os.path.getmtime(previous):
            latest_by_subject[subject_id] = path
    if not latest_by_subject:
        raise FileNotFoundError(f"No subject checkpoint directories found in '{config_dir}'.")
    return OrderedDict(sorted(latest_by_subject.items()))


def _resolve_eval_saw_params(train_cfg, default_saw_shrink):
    train_saw_enabled = _to_bool(train_cfg.get("train_saw", False), False)
    if train_saw_enabled:
        saw_shrink = float(train_cfg.get("train_saw_shrink", default_saw_shrink))
        saw_diag = _to_bool(train_cfg.get("train_saw_diag", False), False)
        saw_renorm = not _to_bool(train_cfg.get("train_saw_no_renorm", False), False)
    else:
        saw_shrink = float(default_saw_shrink)
        saw_diag = False
        saw_renorm = True
    return {
        "train_saw_enabled": train_saw_enabled,
        "saw_shrink": saw_shrink,
        "saw_diag": saw_diag,
        "saw_renorm": saw_renorm,
    }


def _build_shared_split(dataset_size, split_seed, split_a_size=None, split_b_size=None):
    if split_a_size is None and split_b_size is None:
        if dataset_size % 2 != 0:
            raise ValueError(f"Expected an even test set size for default half split, got {dataset_size}.")
        split_a_size = dataset_size // 2
        split_b_size = dataset_size - split_a_size
    elif split_a_size is None:
        split_b_size = int(split_b_size)
        split_a_size = dataset_size - split_b_size
    elif split_b_size is None:
        split_a_size = int(split_a_size)
        split_b_size = dataset_size - split_a_size
    else:
        split_a_size = int(split_a_size)
        split_b_size = int(split_b_size)

    if split_a_size <= 0 or split_b_size <= 0:
        raise ValueError(
            f"Split sizes must both be positive, got split_a_size={split_a_size} split_b_size={split_b_size}."
        )
    if split_a_size + split_b_size != dataset_size:
        raise ValueError(
            f"Split sizes must sum to dataset_size={dataset_size}, got "
            f"split_a_size={split_a_size} split_b_size={split_b_size}."
        )

    rng = np.random.default_rng(split_seed)
    permuted = rng.permutation(np.arange(dataset_size, dtype=np.int64))
    split_a = sorted(permuted[:split_a_size].tolist())
    split_b = sorted(permuted[split_a_size:split_a_size + split_b_size].tolist())
    return split_a, split_b


def _summarize_run(subject_rows, alpha_values):
    train_saw_shrink_values = sorted({
        row["eval_saw_shrink"] if row["train_saw_enabled"] else None
        for row in subject_rows
    }, key=lambda value: (value is not None, value))
    shared = {
        "config": subject_rows[0]["config"],
        "train_saw_shrink_value": (
            None
            if train_saw_shrink_values == [None]
            else ",".join(_format_float_tag(value) for value in train_saw_shrink_values if value is not None)
        ),
        "eval_saw_shrink_values": ",".join(sorted({_format_float_tag(row["eval_saw_shrink"]) for row in subject_rows})),
        "eval_saw_renorm_values": ",".join(sorted({str(row["eval_saw_renorm"]).lower() for row in subject_rows})),
    }

    summary_rows = []
    evaluation_specs = [
        ("cosine_half_b", None, "cosine_half_b_top1_acc", "cosine_half_b_top5_acc"),
        ("saw_cosine_half_b", None, "saw_cosine_half_b_top1_acc", "saw_cosine_half_b_top5_acc"),
    ]
    for alpha in alpha_values:
        alpha_tag = _format_float_tag(alpha)
        evaluation_specs.append(
            (
                f"transfer_alpha_{alpha_tag}_half_b",
                float(alpha),
                f"transfer_alpha_{alpha_tag}_half_b_top1_acc",
                f"transfer_alpha_{alpha_tag}_half_b_top5_acc",
            )
        )

    for evaluation, alpha, top1_column, top5_column in evaluation_specs:
        summary_rows.append({
            **shared,
            "evaluation": evaluation,
            "alpha": alpha,
            "top1_acc": round(float(np.mean([row[top1_column] for row in subject_rows])), 4),
            "top5_acc": round(float(np.mean([row[top5_column] for row in subject_rows])), 4),
        })
    return summary_rows


def _run_subject(config_name, config_dir, checkpoint_dir, runtime_args, split_cache):
    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    subject_ids = train_cfg.get("test_subject_ids", [])
    if len(subject_ids) != 1:
        raise ValueError(f"Expected exactly one test_subject_id in '{checkpoint_dir}', got {subject_ids}.")
    subject_id = int(subject_ids[0])

    eval_args = _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
    test_dataset = _build_test_dataset(eval_args, subject_id)
    dataset_size = len(test_dataset)
    if dataset_size not in split_cache:
        split_cache[dataset_size] = _build_shared_split(
            dataset_size,
            runtime_args.split_seed,
            split_a_size=runtime_args.split_a_size,
            split_b_size=runtime_args.split_b_size,
        )
    split_a_indices, split_b_indices = split_cache[dataset_size]

    modules = _load_modules(eval_args, checkpoint_dir, test_dataset)
    split_a_query, split_a_image = _encode_subset_features(eval_args, modules, test_dataset, split_a_indices)
    split_b_query, split_b_image = _encode_subset_features(eval_args, modules, test_dataset, split_b_indices)

    saw_cfg = _resolve_eval_saw_params(train_cfg, runtime_args.default_saw_shrink)
    saw_metrics = _evaluate_saw_transfer(
        split_a_query,
        split_b_query,
        split_b_image,
        SimpleNamespace(
            saw_shrink=saw_cfg["saw_shrink"],
            saw_diag=saw_cfg["saw_diag"],
            saw_renorm=saw_cfg["saw_renorm"],
        ),
    )
    cosine_metrics = _evaluate_plain(split_b_query, split_b_image)

    transfer_match_eval_saw = _to_bool(getattr(runtime_args, "transfer_match_eval_saw", True), True)
    transfer_params = SimpleNamespace(
        saw_shrink=saw_cfg["saw_shrink"] if transfer_match_eval_saw else runtime_args.transfer_saw_shrink,
        saw_diag=saw_cfg["saw_diag"] if transfer_match_eval_saw else False,
        saw_renorm=saw_cfg["saw_renorm"] if transfer_match_eval_saw else (not runtime_args.transfer_no_renorm),
        use_csls=True,
        csls_k=runtime_args.transfer_csls_k,
        soft_procrustes_enabled=True,
        soft_procrustes_steps=runtime_args.transfer_soft_procrustes_steps,
        soft_procrustes_power=runtime_args.transfer_soft_procrustes_power,
        soft_procrustes_normalize_inputs=False,
        sinkhorn_tau=runtime_args.transfer_sinkhorn_tau,
        sinkhorn_iters=runtime_args.transfer_sinkhorn_iters,
    )
    frozen_calibration = _fit_frozen_calibration(split_a_query, split_a_image, transfer_params)

    row = {
        "config": config_name,
        "config_dir": config_dir,
        "checkpoint_dir": checkpoint_dir,
        "subject_id": subject_id,
        "test_set_size": dataset_size,
        "split_a_size": len(split_a_indices),
        "split_b_size": len(split_b_indices),
        "train_saw_enabled": saw_cfg["train_saw_enabled"],
        "eval_saw_shrink": saw_cfg["saw_shrink"],
        "eval_saw_diag": saw_cfg["saw_diag"],
        "eval_saw_renorm": saw_cfg["saw_renorm"],
        "cosine_half_b_top1_acc": round(float(cosine_metrics["top1_acc"]), 4),
        "cosine_half_b_top5_acc": round(float(cosine_metrics["top5_acc"]), 4),
        "saw_cosine_half_b_top1_acc": round(float(saw_metrics["top1_acc"]), 4),
        "saw_cosine_half_b_top5_acc": round(float(saw_metrics["top5_acc"]), 4),
    }
    for alpha in runtime_args.alpha_values:
        alpha_tag = _format_float_tag(alpha)
        metrics = _evaluate_frozen_transfer_with_alpha(split_b_query, split_b_image, frozen_calibration, alpha)
        row[f"transfer_alpha_{alpha_tag}_half_b_top1_acc"] = round(float(metrics["top1_acc"]), 4)
        row[f"transfer_alpha_{alpha_tag}_half_b_top5_acc"] = round(float(metrics["top5_acc"]), 4)
    return row


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For every config listed in a session_summary.csv, or for one source run directory, split each held-out "
            "test fold in half, evaluate cosine and SAW+cosine on half B, fit a frozen SAATC map on half A, and "
            "sweep alpha on half B."
        )
    )
    parser.add_argument("--session_summary", type=str, default=None)
    parser.add_argument("--source_run_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2099)
    parser.add_argument("--split_seed", type=int, default=2099)
    parser.add_argument("--split_a_size", type=int, default=None)
    parser.add_argument("--split_b_size", type=int, default=None)
    parser.add_argument("--default_saw_shrink", type=float, default=0.85)
    parser.add_argument("--transfer_saw_shrink", type=float, default=0.85)
    parser.add_argument("--transfer_csls_k", type=int, default=1)
    parser.add_argument("--transfer_sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--transfer_sinkhorn_iters", type=int, default=20)
    parser.add_argument("--transfer_soft_procrustes_steps", type=int, default=10)
    parser.add_argument("--transfer_soft_procrustes_power", type=float, default=1.0)
    parser.add_argument("--transfer_no_renorm", action="store_true", default=True)
    parser.add_argument("--transfer_renorm", action="store_false", dest="transfer_no_renorm")
    parser.add_argument("--transfer_match_eval_saw", action="store_true", default=True)
    parser.add_argument("--transfer_use_fixed_saw", action="store_false", dest="transfer_match_eval_saw")
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    args = parser.parse_args()

    if bool(args.session_summary) == bool(args.source_run_dir):
        raise ValueError("Provide exactly one of --session_summary or --source_run_dir.")

    session_summary = None
    session_dir = None
    config_entries = []
    if args.session_summary is not None:
        session_summary = os.path.abspath(args.session_summary)
        if not os.path.isfile(session_summary):
            raise FileNotFoundError(f"session_summary does not exist: '{session_summary}'")

        session_dir = os.path.dirname(session_summary)
        session_df = pd.read_csv(session_summary)
        if "config" not in session_df.columns:
            raise ValueError(f"'config' column not found in '{session_summary}'")

        configs = [config for config in pd.unique(session_df["config"]) if isinstance(config, str) and config.strip()]
        if not configs:
            raise ValueError(f"No configs found in '{session_summary}'")
        config_entries = [(config_name, os.path.join(session_dir, config_name)) for config_name in configs]
    else:
        source_run_dir = os.path.abspath(args.source_run_dir)
        if not os.path.isdir(source_run_dir):
            raise FileNotFoundError(f"source_run_dir does not exist: '{source_run_dir}'")
        config_name = os.path.basename(os.path.normpath(source_run_dir))
        config_entries = [(config_name, source_run_dir)]

    if args.output_dir is None:
        run_prefix = "session_split_transfer" if args.session_summary is not None else "source_run_split_transfer"
        run_tag = f"{run_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = os.path.join(REPO_ROOT, "results", "things_eeg", "inter-subjects", run_tag)
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)
    split_cache = {}
    subject_rows = []
    run_rows = []

    for config_name, config_dir in config_entries:
        checkpoint_dirs = _resolve_subject_checkpoint_dirs(config_dir)
        config_subject_rows = []
        print(f"[session_split_transfer] config={config_name} subjects={list(checkpoint_dirs.keys())}")
        for subject_id, checkpoint_dir in checkpoint_dirs.items():
            print(f"  subject={subject_id:02d} checkpoint_dir={checkpoint_dir}")
            row = _run_subject(config_name, config_dir, checkpoint_dir, args, split_cache)
            config_subject_rows.append(row)
            subject_rows.append(row)
        run_rows.extend(_summarize_run(config_subject_rows, args.alpha_values))

    pd.DataFrame(subject_rows).to_csv(os.path.join(output_dir, "all_subject_results.csv"), index=False)
    pd.DataFrame(run_rows).to_csv(os.path.join(output_dir, "run_average_results.csv"), index=False)

    split_manifest = []
    for dataset_size in sorted(split_cache):
        split_a_indices, split_b_indices = split_cache[dataset_size]
        split_manifest.append({
            "dataset_size": int(dataset_size),
            "split_seed": int(args.split_seed),
            "split_a_size": len(split_a_indices),
            "split_b_size": len(split_b_indices),
            "split_a_indices": split_a_indices,
            "split_b_indices": split_b_indices,
        })
    with open(os.path.join(output_dir, "shared_split_manifest.json"), "w") as f:
        json.dump(split_manifest, f, indent=4)

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "session_summary": session_summary,
                "session_dir": session_dir,
                "source_run_dir": (
                    os.path.abspath(args.source_run_dir) if args.source_run_dir is not None else None
                ),
                "transfer_param_tag": (
                    "match_eval_saw_k1_tau0p1_steps10_pow1p0_iters20"
                    if args.transfer_match_eval_saw
                    else "saw0p85_k1_tau0p1_steps10_pow1p0_iters20"
                ),
            },
            f,
            indent=4,
        )

    print("\n=== Run Average Results ===")
    print(pd.DataFrame(run_rows).to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
