#!/usr/bin/env python3
"""Disjoint transfer curves for frozen SATTC-style calibration.

For each held-out subject and for fit sizes n = 5, 10, ..., 195:
1. randomly subsample n query/image pairs, apply SAW to the EEG side, and fit
   a supervised orthogonal Procrustes map using the known query/image pairs;
2. freeze the learned correction;
3. evaluate it on the remaining 200-n held-out pairs under two protocols:
   - reduced-set transfer: held-out queries vs held-out candidates only
   - full-set transfer: held-out queries vs full 200 candidates
4. compare to plain cosine on the same retrieval problem.

The script writes raw per-subject results, aggregate CSVs, and four summary
figures: one for each {reduced, full} x {alpha=0.5, alpha=1.0} setting.

Normalized gain is defined as:
    100 * (tta_acc - cosine_acc) / max(cosine_acc, eps)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, topk
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything


def _progress(iterable=None, total=None, desc=None, position=0, leave=True):
    if tqdm is None:
        return iterable if iterable is not None else range(total or 0)
    return tqdm(iterable, total=total, desc=desc, position=position, leave=leave, dynamic_ncols=True)


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


def _fit_paired_procrustes(query_features, image_features, normalize_inputs=False):
    query_features = np.asarray(query_features, dtype=np.float32)
    image_features = np.asarray(image_features, dtype=np.float32)
    if query_features.ndim != 2 or image_features.ndim != 2:
        return None
    if query_features.shape != image_features.shape or query_features.shape[0] == 0:
        return None
    assignment = np.eye(query_features.shape[0], dtype=np.float32)
    return fit_soft_assignment_procrustes(
        query_features,
        image_features,
        assignment,
        power=1.0,
        normalize_inputs=normalize_inputs,
    )


def _fit_frozen_calibration(query_features, image_features, params):
    transformed, saw_stats = _fit_saw_transform(
        query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
    cumulative_map = np.eye(transformed.shape[1], dtype=np.float32)

    if params.soft_procrustes_enabled:
        step_map = _fit_paired_procrustes(
            transformed,
            image_features,
            normalize_inputs=params.soft_procrustes_normalize_inputs,
        )
        if step_map is not None:
            transformed = apply_orthogonal_map(transformed, step_map)
            cumulative_map = (cumulative_map @ np.asarray(step_map, dtype=np.float32)).astype(np.float32, copy=False)

    return {
        "saw_stats": saw_stats,
        "orthogonal_map": cumulative_map,
    }


def _apply_frozen_calibration(features, calibration, alpha):
    transformed = _apply_saw_transform(features, calibration["saw_stats"])
    blended_map = _blend_orthogonal_map(calibration["orthogonal_map"], alpha)
    transformed = apply_orthogonal_map(transformed, blended_map)
    return np.asarray(transformed, dtype=np.float32)


def _apply_raw_procrustes_calibration(features, orthogonal_map, alpha):
    blended_map = _blend_orthogonal_map(orthogonal_map, alpha)
    transformed = apply_orthogonal_map(features, blended_map)
    return np.asarray(transformed, dtype=np.float32)


def _evaluate_scores(scores, target_indices):
    top5_count, top1_count = topk(scores, 5, target_indices=target_indices)
    total = scores.shape[0]
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _find_checkpoint_dir(source_run_dir, subject_id):
    suffix = f"-sub-{int(subject_id):02d}"
    matches = [
        os.path.join(source_run_dir, name)
        for name in os.listdir(source_run_dir)
        if name.endswith(suffix) and os.path.isdir(os.path.join(source_run_dir, name))
    ]
    if not matches:
        for root, dirs, _ in os.walk(source_run_dir):
            for dirname in dirs:
                if dirname.endswith(suffix):
                    matches.append(os.path.join(root, dirname))
    if not matches:
        raise FileNotFoundError(f"Could not find checkpoint directory for subject {subject_id} in '{source_run_dir}'.")
    return max(matches, key=os.path.getmtime)


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
        "device": device,
    }


def _encode_full_subject_features(eval_args, modules, dataset):
    loader = DataLoader(
        dataset,
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
            eeg_feature_batch = modules["eeg_projector"](eeg_backbone_batch)
            image_feature_proj = modules["img_projector"](image_feature_batch)

            eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
            image_feature_list.append(image_feature_proj.cpu().numpy())

    return (
        np.concatenate(eeg_feature_list, axis=0).astype(np.float32, copy=False),
        np.concatenate(image_feature_list, axis=0).astype(np.float32, copy=False),
    )


def _encode_all_subjects(source_run_dir, subject_ids, runtime_args):
    encoded = {}
    progress = _progress(subject_ids, desc="Encoding subjects", position=0, leave=True)
    for subject_id in progress:
        checkpoint_dir = _find_checkpoint_dir(source_run_dir, subject_id)
        train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
        eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
        eval_args = _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
        test_dataset = _build_test_dataset(eval_args, subject_id)
        modules = _load_modules(eval_args, checkpoint_dir, test_dataset)
        eeg_features, image_features = _encode_full_subject_features(eval_args, modules, test_dataset)
        encoded[int(subject_id)] = {
            "checkpoint_dir": checkpoint_dir,
            "eeg_features": eeg_features,
            "image_features": image_features,
        }
        if tqdm is not None:
            progress.set_postfix(subject=f"{int(subject_id):02d}", refresh=False)
    return encoded


def _compute_gains(df, eps=1e-8):
    out = df.copy()
    for metric in ("top1", "top5"):
        tta_col = f"tta_{metric}_acc"
        cosine_col = f"cosine_{metric}_acc"
        pure_col = f"pure_cosine_{metric}_acc"
        if pure_col in out.columns:
            base_values = out[pure_col].where(out[pure_col].notna(), out[cosine_col])
        else:
            base_values = out[cosine_col]
        out[f"absolute_gain_{metric}"] = out[tta_col] - base_values
        out[f"normalized_gain_{metric}"] = 100.0 * (out[tta_col] - base_values) / np.clip(base_values, eps, None)
    return out


RAW_RESULT_COLUMNS = [
    "subject_id",
    "repeat",
    "fit_size",
    "alpha",
    "scope",
    "tta_top1_acc",
    "tta_top5_acc",
    "cosine_top1_acc",
    "cosine_top5_acc",
    "pure_cosine_top1_acc",
    "pure_cosine_top5_acc",
    "procrustes_only_top1_acc",
    "procrustes_only_top5_acc",
    "no_saw_tta_top1_acc",
    "no_saw_tta_top5_acc",
    "saw_shrink_0p2_top1_acc",
    "saw_shrink_0p2_top5_acc",
    "saw_shrink_0p4_top1_acc",
    "saw_shrink_0p4_top5_acc",
    "saw_shrink_0p6_top1_acc",
    "saw_shrink_0p6_top5_acc",
    "saw_shrink_0p8_top1_acc",
    "saw_shrink_0p8_top5_acc",
    "absolute_gain_top1",
    "normalized_gain_top1",
    "absolute_gain_top5",
    "normalized_gain_top5",
]
RAW_KEY_COLUMNS = ["subject_id", "repeat", "fit_size", "alpha", "scope"]
TRANSFER_SCOPES = ("reduced_set_transfer", "full_set_transfer")
PLOT_SAW_SHRINKS = (0.2, 0.4, 0.6, 0.8)


def _saw_shrink_tag(shrink):
    return str(shrink).replace(".", "p")


def _alpha_key(alpha):
    return format(float(alpha), ".12g")


def _standardize_raw_results(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_RESULT_COLUMNS)

    out = df.copy()
    optional_columns = [
        "pure_cosine_top1_acc",
        "pure_cosine_top5_acc",
        "procrustes_only_top1_acc",
        "procrustes_only_top5_acc",
        "no_saw_tta_top1_acc",
        "no_saw_tta_top5_acc",
        *[
            f"saw_shrink_{_saw_shrink_tag(shrink)}_{metric}_acc"
            for shrink in PLOT_SAW_SHRINKS
            for metric in ("top1", "top5")
        ],
    ]
    for col in optional_columns:
        if col not in out.columns:
            out[col] = np.nan
    if any(col not in out.columns for col in ("absolute_gain_top1", "normalized_gain_top1", "absolute_gain_top5", "normalized_gain_top5")):
        out = _compute_gains(out)

    missing = [col for col in RAW_RESULT_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Raw results are missing required columns: {missing}")

    out = out[RAW_RESULT_COLUMNS].copy()
    out["subject_id"] = out["subject_id"].astype(int)
    out["repeat"] = out["repeat"].astype(int)
    out["fit_size"] = out["fit_size"].astype(int)
    out["alpha"] = out["alpha"].astype(float)
    out["scope"] = out["scope"].astype(str)
    out = out.drop_duplicates(subset=RAW_KEY_COLUMNS, keep="last")
    out = out.sort_values(["subject_id", "repeat", "fit_size", "alpha", "scope"]).reset_index(drop=True)
    return out


def _load_existing_raw_results(raw_path):
    if not os.path.isfile(raw_path):
        return pd.DataFrame(columns=RAW_RESULT_COLUMNS)

    raw_df = pd.read_csv(raw_path)
    raw_df = _standardize_raw_results(raw_df)
    raw_df.to_csv(raw_path, index=False)
    return raw_df


def _append_raw_results(raw_path, rows):
    if not rows:
        return

    batch_df = _standardize_raw_results(pd.DataFrame(rows))
    write_header = not os.path.isfile(raw_path) or os.path.getsize(raw_path) == 0
    batch_df.to_csv(raw_path, mode="a", header=write_header, index=False)


def _filter_requested_results(raw_df, subjects, num_repeats, fit_sizes, alpha_values):
    if raw_df.empty:
        return raw_df.copy()

    alpha_keys = {_alpha_key(alpha) for alpha in alpha_values}
    filtered = raw_df[
        raw_df["subject_id"].isin([int(subject_id) for subject_id in subjects])
        & raw_df["repeat"].isin(list(range(int(num_repeats))))
        & raw_df["fit_size"].isin([int(fit_size) for fit_size in fit_sizes])
        & raw_df["scope"].isin(list(TRANSFER_SCOPES))
    ].copy()
    filtered = filtered[filtered["alpha"].map(_alpha_key).isin(alpha_keys)]
    return _standardize_raw_results(filtered)


def _build_existing_row_keys(raw_df):
    if raw_df.empty:
        return set()

    return {
        (int(row.subject_id), int(row.repeat), int(row.fit_size), _alpha_key(row.alpha), str(row.scope))
        for row in raw_df.itertuples(index=False)
        if not hasattr(row, "no_saw_tta_top1_acc") or pd.notna(row.no_saw_tta_top1_acc)
    }


def _fit_size_row_keys(subject_id, repeat_idx, fit_size, alpha_values):
    return {
        (int(subject_id), int(repeat_idx), int(fit_size), _alpha_key(alpha), scope)
        for alpha in alpha_values
        for scope in TRANSFER_SCOPES
    }


def _aggregate_results(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "alpha",
                "fit_size",
                "tta_top1_mean",
                "tta_top1_std",
                "tta_top5_mean",
                "tta_top5_std",
                "cosine_top1_mean",
                "cosine_top1_std",
                "cosine_top5_mean",
                "cosine_top5_std",
                "pure_cosine_top1_mean",
                "pure_cosine_top1_std",
                "pure_cosine_top5_mean",
                "pure_cosine_top5_std",
                "no_saw_tta_top1_mean",
                "no_saw_tta_top1_std",
                "saw_shrink_0p2_top1_mean",
                "saw_shrink_0p2_top1_std",
                "saw_shrink_0p4_top1_mean",
                "saw_shrink_0p4_top1_std",
                "saw_shrink_0p6_top1_mean",
                "saw_shrink_0p6_top1_std",
                "saw_shrink_0p8_top1_mean",
                "saw_shrink_0p8_top1_std",
                "absolute_gain_top1_mean",
                "absolute_gain_top1_std",
                "absolute_gain_top5_mean",
                "absolute_gain_top5_std",
                "normalized_gain_top1_mean",
                "normalized_gain_top1_std",
                "normalized_gain_top5_mean",
                "normalized_gain_top5_std",
            ]
        )

    return (
        df.groupby(["scope", "alpha", "fit_size"], as_index=False)
        .agg(
            tta_top1_mean=("tta_top1_acc", "mean"),
            tta_top1_std=("tta_top1_acc", "std"),
            tta_top5_mean=("tta_top5_acc", "mean"),
            tta_top5_std=("tta_top5_acc", "std"),
            cosine_top1_mean=("cosine_top1_acc", "mean"),
            cosine_top1_std=("cosine_top1_acc", "std"),
            cosine_top5_mean=("cosine_top5_acc", "mean"),
            cosine_top5_std=("cosine_top5_acc", "std"),
            pure_cosine_top1_mean=("pure_cosine_top1_acc", "mean"),
            pure_cosine_top1_std=("pure_cosine_top1_acc", "std"),
            pure_cosine_top5_mean=("pure_cosine_top5_acc", "mean"),
            pure_cosine_top5_std=("pure_cosine_top5_acc", "std"),
            no_saw_tta_top1_mean=("no_saw_tta_top1_acc", "mean"),
            no_saw_tta_top1_std=("no_saw_tta_top1_acc", "std"),
            saw_shrink_0p2_top1_mean=("saw_shrink_0p2_top1_acc", "mean"),
            saw_shrink_0p2_top1_std=("saw_shrink_0p2_top1_acc", "std"),
            saw_shrink_0p4_top1_mean=("saw_shrink_0p4_top1_acc", "mean"),
            saw_shrink_0p4_top1_std=("saw_shrink_0p4_top1_acc", "std"),
            saw_shrink_0p6_top1_mean=("saw_shrink_0p6_top1_acc", "mean"),
            saw_shrink_0p6_top1_std=("saw_shrink_0p6_top1_acc", "std"),
            saw_shrink_0p8_top1_mean=("saw_shrink_0p8_top1_acc", "mean"),
            saw_shrink_0p8_top1_std=("saw_shrink_0p8_top1_acc", "std"),
            absolute_gain_top1_mean=("absolute_gain_top1", "mean"),
            absolute_gain_top1_std=("absolute_gain_top1", "std"),
            absolute_gain_top5_mean=("absolute_gain_top5", "mean"),
            absolute_gain_top5_std=("absolute_gain_top5", "std"),
            normalized_gain_top1_mean=("normalized_gain_top1", "mean"),
            normalized_gain_top1_std=("normalized_gain_top1", "std"),
            normalized_gain_top5_mean=("normalized_gain_top5", "mean"),
            normalized_gain_top5_std=("normalized_gain_top5", "std"),
        )
        .fillna(0.0)
    )


def _plot_gain_panel(ax, x, alpha_results, ylabel, title):
    # alpha_results is a list of (alpha, mean, std)
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_results)))
    for (alpha, mean, std), color in zip(alpha_results, colors):
        label = f"alpha={alpha:g}"
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Fit size n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_scope_multi_alpha(agg_df, scope, alpha_values, output_dir):
    scope_df = agg_df[agg_df["scope"] == scope]
    if scope_df.empty:
        return

    # Get common x-axis (fit_sizes)
    fit_sizes = sorted(scope_df["fit_size"].unique())
    x = np.array(fit_sizes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=220)

    # Extract baseline data
    first_alpha = sorted(alpha_values)[0]
    alpha_df_first = scope_df[np.isclose(scope_df["alpha"], first_alpha)].sort_values("fit_size")
    if alpha_df_first.empty:
        plt.close(fig)
        return
    
    alpha_df_first = alpha_df_first.set_index("fit_size").reindex(fit_sizes).reset_index()
    
    pure_mean = alpha_df_first["pure_cosine_top1_mean"].to_numpy()
    saw_mean = alpha_df_first["cosine_top1_mean"].to_numpy()

    # Top row: with SAW
    axes[0, 0].plot(x, saw_mean - pure_mean, color="black", linewidth=1.5, label="cos+saw 0.94")
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    for alpha, color in zip(sorted(alpha_values), colors):
        alpha_df = scope_df[np.isclose(scope_df["alpha"], alpha)].sort_values("fit_size")
        if alpha_df.empty:
            continue
        alpha_df = alpha_df.set_index("fit_size").reindex(fit_sizes).reset_index()
        tta_mean = alpha_df["tta_top1_mean"].to_numpy()
        axes[0, 0].plot(x, tta_mean - pure_mean, color=color, linewidth=2.0, label=f"alpha={alpha:g}")

    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0, 0].set_xlabel("Fit size n")
    axes[0, 0].set_ylabel("Absolute gain (points)")
    axes[0, 0].set_title("Absolute gain (with SAW)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, pure_mean, color="black", linestyle="--", linewidth=1.2, label="pure cosine")
    axes[0, 1].plot(x, saw_mean, color="black", linewidth=1.5, label="cos+saw 0.94")
    for alpha, color in zip(sorted(alpha_values), colors):
        alpha_df = scope_df[np.isclose(scope_df["alpha"], alpha)].sort_values("fit_size")
        if alpha_df.empty:
            continue
        alpha_df = alpha_df.set_index("fit_size").reindex(fit_sizes).reset_index()
        tta_mean = alpha_df["tta_top1_mean"].to_numpy()
        axes[0, 1].plot(x, tta_mean, color=color, linewidth=2.0, label=f"alpha={alpha:g}")

    axes[0, 1].set_xlabel("Fit size n")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Raw Top-1 Accuracy (with SAW)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom row: no SAW
    axes[1, 0].plot(x, saw_mean - pure_mean, color="black", linewidth=1.5, label="cos+saw 0.94")
    for alpha, color in zip(sorted(alpha_values), colors):
        alpha_df = scope_df[np.isclose(scope_df["alpha"], alpha)].sort_values("fit_size")
        if alpha_df.empty:
            continue
        alpha_df = alpha_df.set_index("fit_size").reindex(fit_sizes).reset_index()
        no_saw_mean = alpha_df["no_saw_tta_top1_mean"].to_numpy()
        axes[1, 0].plot(x, no_saw_mean - pure_mean, color=color, linewidth=2.0, label=f"alpha={alpha:g}")

    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1, 0].set_xlabel("Fit size n")
    axes[1, 0].set_ylabel("Absolute gain (points)")
    axes[1, 0].set_title("Absolute gain (no SAW)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, pure_mean, color="black", linestyle="--", linewidth=1.2, label="pure cosine")
    axes[1, 1].plot(x, saw_mean, color="black", linewidth=1.5, label="cos+saw 0.94")
    for alpha, color in zip(sorted(alpha_values), colors):
        alpha_df = scope_df[np.isclose(scope_df["alpha"], alpha)].sort_values("fit_size")
        if alpha_df.empty:
            continue
        alpha_df = alpha_df.set_index("fit_size").reindex(fit_sizes).reset_index()
        no_saw_mean = alpha_df["no_saw_tta_top1_mean"].to_numpy()
        axes[1, 1].plot(x, no_saw_mean, color=color, linewidth=2.0, label=f"alpha={alpha:g}")

    axes[1, 1].set_xlabel("Fit size n")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Raw Top-1 Accuracy (no SAW)")
    axes[1, 1].grid(True, alpha=0.3)

    # Legend and layout
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{scope.replace('_', ' ').title()} transfer (Top-1)", y=1.03)
    fig.tight_layout()
    filename = f"{scope}_multi_alpha_gain_plot.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Disjoint held-out transfer curves for frozen SATTC-style TTA."
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--fit_sizes", nargs="+", type=int, default=list(range(5, 200, 5)))
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.94)
    parser.add_argument("--sattc_csls_k", type=int, default=3)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=12)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=16)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.2)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: '{source_run_dir}'")

    if args.output_dir is None:
        run_tag = f"disjoint_tta_transfer_curve_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = os.path.join(REPO_ROOT, "results", "things_eeg", "inter-subjects", run_tag)
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "disjoint_transfer_raw_results.csv")
    agg_path = os.path.join(output_dir, "disjoint_transfer_aggregate_results.csv")
    summary_path = os.path.join(output_dir, "disjoint_transfer_summary.csv")

    seed_everything(args.seed)

    fit_sizes = sorted({int(size) for size in args.fit_sizes if 0 < int(size) < 200})
    alpha_values = sorted({float(alpha) for alpha in args.alpha_values})
    existing_raw_df = _filter_requested_results(
        _load_existing_raw_results(raw_path),
        args.subjects,
        args.num_repeats,
        fit_sizes,
        alpha_values,
    )
    existing_row_keys = _build_existing_row_keys(existing_raw_df)

    params = SimpleNamespace(
        saw_shrink=args.sattc_saw_shrink,
        saw_diag=bool(args.sattc_saw_diag),
        saw_renorm=not bool(args.sattc_saw_no_renorm),
        use_csls=True,
        csls_k=args.sattc_csls_k,
        soft_procrustes_enabled=True,
        soft_procrustes_steps=args.sattc_soft_procrustes_steps,
        soft_procrustes_power=args.sattc_soft_procrustes_power,
        soft_procrustes_normalize_inputs=bool(args.sattc_soft_procrustes_normalize_inputs),
        sinkhorn_tau=args.sattc_sinkhorn_tau,
        sinkhorn_iters=args.sattc_sinkhorn_iters,
    )

    pending_subjects = [
        int(subject_id)
        for subject_id in args.subjects
        if any(
            not _fit_size_row_keys(subject_id, repeat_idx, fit_size, alpha_values).issubset(existing_row_keys)
            for repeat_idx in range(args.num_repeats)
            for fit_size in fit_sizes
        )
    ]

    encoded = {}
    if pending_subjects:
        print(f"Resuming from {raw_path} and encoding only pending subjects...")
        encoded = _encode_all_subjects(source_run_dir, pending_subjects, args)
    else:
        print(f"All requested raw results already exist in {raw_path}; skipping feature encoding and fit-time computation.")

    subject_progress = _progress(pending_subjects, desc="Subjects", position=0, leave=True)
    for subject_id in subject_progress:
        subject_id = int(subject_id)
        eeg_features = encoded[subject_id]["eeg_features"]
        image_features = encoded[subject_id]["image_features"]
        total = eeg_features.shape[0]
        rng = np.random.default_rng(args.seed + subject_id)

        if tqdm is not None:
            subject_progress.set_postfix(subject=f"{subject_id:02d}", refresh=False)

        repeat_progress = _progress(
            range(args.num_repeats),
            desc=f"sub-{subject_id:02d} repeats",
            position=1,
            leave=False,
        )
        for repeat_idx in repeat_progress:
            perm = rng.permutation(total)
            fit_progress = _progress(
                fit_sizes,
                desc=f"sub-{subject_id:02d} fit sizes",
                position=2,
                leave=False,
            )
            for fit_size in fit_progress:
                fit_row_keys = _fit_size_row_keys(subject_id, repeat_idx, fit_size, alpha_values)
                if fit_row_keys.issubset(existing_row_keys):
                    if tqdm is not None:
                        fit_progress.set_postfix(fit=fit_size, holdout=max(total - fit_size, 0), status="resume", refresh=False)
                    continue

                fit_idx = np.sort(perm[:fit_size])
                holdout_idx = np.sort(perm[fit_size:])

                # Reduced set experiment: evaluate only on the remaining 200-n samples
                # If n=200, there are no samples to evaluate, so we skip.
                if holdout_idx.size == 0:
                    continue

                calibration = _fit_frozen_calibration(eeg_features[fit_idx], image_features[fit_idx], params)

                eval_queries = eeg_features[holdout_idx]
                
                # Define evaluation targets/candidates
                reduced_candidates = image_features[holdout_idx]
                reduced_targets = np.arange(holdout_idx.size, dtype=np.int64)

                # 1. Pure Cosine (No SAW, No Procrustes)
                cosine_pure_reduced = _evaluate_scores(_cosine_scores(eval_queries, reduced_candidates), reduced_targets)
                cosine_pure_full = _evaluate_scores(_cosine_scores(eval_queries, image_features), holdout_idx.astype(np.int64))

                # 1b. Procrustes only (No SAW)
                procrustes_only_map = _fit_paired_procrustes(
                    eeg_features[fit_idx],
                    image_features[fit_idx],
                    normalize_inputs=params.soft_procrustes_normalize_inputs,
                )
                if procrustes_only_map is None:
                    transformed_eval_procrustes_only = np.asarray(eval_queries, dtype=np.float32)
                else:
                    transformed_eval_procrustes_only = apply_orthogonal_map(eval_queries, procrustes_only_map)
                procrustes_only_reduced = _evaluate_scores(
                    _cosine_scores(transformed_eval_procrustes_only, reduced_candidates),
                    reduced_targets,
                )
                procrustes_only_full = _evaluate_scores(
                    _cosine_scores(transformed_eval_procrustes_only, image_features),
                    holdout_idx.astype(np.int64),
                )

                # 2. SAW-only baselines at multiple shrinkages
                saw_only_scores = {}
                for shrink in PLOT_SAW_SHRINKS:
                    _, saw_stats = _fit_saw_transform(
                        eeg_features[fit_idx],
                        shrink=shrink,
                        diag=params.saw_diag,
                        normalize=params.saw_renorm,
                    )
                    transformed_eval_saw = _apply_saw_transform(eval_queries, saw_stats)
                    saw_only_scores[shrink] = {
                        "reduced": _evaluate_scores(_cosine_scores(transformed_eval_saw, reduced_candidates), reduced_targets),
                        "full": _evaluate_scores(_cosine_scores(transformed_eval_saw, image_features), holdout_idx.astype(np.int64)),
                    }

                # Keep the configured shrink baseline for backward-compatible CSV columns.
                transformed_eval_baseline = _apply_frozen_calibration(
                    eval_queries,
                    {"saw_stats": calibration["saw_stats"], "orthogonal_map": np.eye(eeg_features.shape[1], dtype=np.float32)},
                    0.0,
                )
                cosine_reduced = _evaluate_scores(_cosine_scores(transformed_eval_baseline, reduced_candidates), reduced_targets)
                cosine_full = _evaluate_scores(_cosine_scores(transformed_eval_baseline, image_features), holdout_idx.astype(np.int64))

                fit_rows = []
                for alpha in alpha_values:
                    # TTA: SAW + Procrustes (Alpha > 0)
                    transformed_eval_tta = _apply_frozen_calibration(eval_queries, calibration, alpha)
                    raw_map = procrustes_only_map if procrustes_only_map is not None else np.eye(eeg_features.shape[1], dtype=np.float32)
                    transformed_eval_tta_no_saw = _apply_raw_procrustes_calibration(eval_queries, raw_map, alpha)
                    tta_reduced = _evaluate_scores(_cosine_scores(transformed_eval_tta, reduced_candidates), reduced_targets)
                    tta_full = _evaluate_scores(_cosine_scores(transformed_eval_tta, image_features), holdout_idx.astype(np.int64))
                    tta_no_saw_reduced = _evaluate_scores(_cosine_scores(transformed_eval_tta_no_saw, reduced_candidates), reduced_targets)
                    tta_no_saw_full = _evaluate_scores(_cosine_scores(transformed_eval_tta_no_saw, image_features), holdout_idx.astype(np.int64))

                    fit_rows.extend(
                        [
                            {
                                "subject_id": subject_id,
                                "repeat": repeat_idx,
                                "fit_size": fit_size,
                                "alpha": alpha,
                                "scope": "reduced_set_transfer",
                                "tta_top1_acc": round(float(tta_reduced["top1_acc"]), 4),
                                "tta_top5_acc": round(float(tta_reduced["top5_acc"]), 4),
                                "cosine_top1_acc": round(float(cosine_reduced["top1_acc"]), 4),
                                "cosine_top5_acc": round(float(cosine_reduced["top5_acc"]), 4),
                                "pure_cosine_top1_acc": round(float(cosine_pure_reduced["top1_acc"]), 4),
                                "pure_cosine_top5_acc": round(float(cosine_pure_reduced["top5_acc"]), 4),
                                "procrustes_only_top1_acc": round(float(procrustes_only_reduced["top1_acc"]), 4),
                                "procrustes_only_top5_acc": round(float(procrustes_only_reduced["top5_acc"]), 4),
                                "no_saw_tta_top1_acc": round(float(tta_no_saw_reduced["top1_acc"]), 4),
                                "no_saw_tta_top5_acc": round(float(tta_no_saw_reduced["top5_acc"]), 4),
                                "saw_shrink_0p2_top1_acc": round(float(saw_only_scores[0.2]["reduced"]["top1_acc"]), 4),
                                "saw_shrink_0p2_top5_acc": round(float(saw_only_scores[0.2]["reduced"]["top5_acc"]), 4),
                                "saw_shrink_0p4_top1_acc": round(float(saw_only_scores[0.4]["reduced"]["top1_acc"]), 4),
                                "saw_shrink_0p4_top5_acc": round(float(saw_only_scores[0.4]["reduced"]["top5_acc"]), 4),
                                "saw_shrink_0p6_top1_acc": round(float(saw_only_scores[0.6]["reduced"]["top1_acc"]), 4),
                                "saw_shrink_0p6_top5_acc": round(float(saw_only_scores[0.6]["reduced"]["top5_acc"]), 4),
                                "saw_shrink_0p8_top1_acc": round(float(saw_only_scores[0.8]["reduced"]["top1_acc"]), 4),
                                "saw_shrink_0p8_top5_acc": round(float(saw_only_scores[0.8]["reduced"]["top5_acc"]), 4),
                            },
                            {
                                "subject_id": subject_id,
                                "repeat": repeat_idx,
                                "fit_size": fit_size,
                                "alpha": alpha,
                                "scope": "full_set_transfer",
                                "tta_top1_acc": round(float(tta_full["top1_acc"]), 4),
                                "tta_top5_acc": round(float(tta_full["top5_acc"]), 4),
                                "cosine_top1_acc": round(float(cosine_full["top1_acc"]), 4),
                                "cosine_top5_acc": round(float(cosine_full["top5_acc"]), 4),
                                "pure_cosine_top1_acc": round(float(cosine_pure_full["top1_acc"]), 4),
                                "pure_cosine_top5_acc": round(float(cosine_pure_full["top5_acc"]), 4),
                                "procrustes_only_top1_acc": round(float(procrustes_only_full["top1_acc"]), 4),
                                "procrustes_only_top5_acc": round(float(procrustes_only_full["top5_acc"]), 4),
                                "no_saw_tta_top1_acc": round(float(tta_no_saw_full["top1_acc"]), 4),
                                "no_saw_tta_top5_acc": round(float(tta_no_saw_full["top5_acc"]), 4),
                                "saw_shrink_0p2_top1_acc": round(float(saw_only_scores[0.2]["full"]["top1_acc"]), 4),
                                "saw_shrink_0p2_top5_acc": round(float(saw_only_scores[0.2]["full"]["top5_acc"]), 4),
                                "saw_shrink_0p4_top1_acc": round(float(saw_only_scores[0.4]["full"]["top1_acc"]), 4),
                                "saw_shrink_0p4_top5_acc": round(float(saw_only_scores[0.4]["full"]["top5_acc"]), 4),
                                "saw_shrink_0p6_top1_acc": round(float(saw_only_scores[0.6]["full"]["top1_acc"]), 4),
                                "saw_shrink_0p6_top5_acc": round(float(saw_only_scores[0.6]["full"]["top5_acc"]), 4),
                                "saw_shrink_0p8_top1_acc": round(float(saw_only_scores[0.8]["full"]["top1_acc"]), 4),
                                "saw_shrink_0p8_top5_acc": round(float(saw_only_scores[0.8]["full"]["top5_acc"]), 4),
                            },
                        ]
                    )
                fit_rows = _compute_gains(pd.DataFrame(fit_rows)).to_dict("records")
                _append_raw_results(raw_path, fit_rows)
                existing_row_keys.update(fit_row_keys)
                if tqdm is not None:
                    fit_progress.set_postfix(
                        fit=fit_size,
                        holdout=int(holdout_idx.size),
                        status="saved",
                        refresh=False,
                    )
            if tqdm is not None:
                repeat_progress.set_postfix(repeat=repeat_idx + 1, refresh=False)

    raw_df = _filter_requested_results(
        _load_existing_raw_results(raw_path),
        args.subjects,
        args.num_repeats,
        fit_sizes,
        alpha_values,
    )
    agg_df = _aggregate_results(raw_df)
    agg_df.to_csv(agg_path, index=False)

    summary_rows = []
    for scope in sorted(agg_df["scope"].unique().tolist()):
        for alpha in alpha_values:
            alpha_df = agg_df[(agg_df["scope"] == scope) & (np.isclose(agg_df["alpha"], alpha))]
            if alpha_df.empty:
                continue
            best_row = alpha_df.sort_values("absolute_gain_top1_mean", ascending=False).iloc[0]
            summary_rows.append(
                {
                    "scope": scope,
                    "alpha": alpha,
                    "best_fit_size_by_top1_gain": int(best_row["fit_size"]),
                    "best_absolute_gain_top1_mean": float(best_row["absolute_gain_top1_mean"]),
                    "best_normalized_gain_top1_mean": float(best_row["normalized_gain_top1_mean"]),
                    "best_absolute_gain_top5_mean": float(best_row["absolute_gain_top5_mean"]),
                    "best_normalized_gain_top5_mean": float(best_row["normalized_gain_top5_mean"]),
                }
            )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    for scope in TRANSFER_SCOPES:
        _plot_scope_multi_alpha(agg_df, scope, alpha_values, output_dir)

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "source_run_dir": source_run_dir,
                "resolved_fit_sizes": fit_sizes,
                "resolved_alpha_values": alpha_values,
            },
            f,
            indent=4,
        )

    print(f"\nRaw results: {raw_path}")
    print(f"Aggregate results: {agg_path}")
    print(f"Summary: {summary_path}")
    print(f"Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
