#!/usr/bin/env python3
"""Run calibration-style SATTC transfer experiments with fixed TTA logic.

This script focuses on the most paper-relevant "more inductive" variants that
stay close to the current SAW + CSLS + Sinkhorn + soft-Procrustes pipeline:

1. Full 200-way baselines on each held-out subject:
   - plain cosine
   - SAW only
   - fresh full SATTC
2. Progressive frozen calibration transfer:
   - fit SAW + soft-Procrustes on K unlabeled query/image pairs
   - freeze the learned transform
   - apply it to:
       a) the full 200-query set, scored against the full 200 image set
       b) only the held-out remainder queries, still scored against the full
          200 image set
   - repeat for K = 5, 10, ..., 200 and for several alpha blends between
     identity and the learned orthogonal map.

Outputs are written as CSV summaries and a few lightweight plots.
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


def _apply_frozen_calibration(features, calibration, alpha):
    transformed = _apply_saw_transform(features, calibration["saw_stats"])
    blended_map = _blend_orthogonal_map(calibration["orthogonal_map"], alpha)
    transformed = apply_orthogonal_map(transformed, blended_map)
    return np.asarray(transformed, dtype=np.float32)


def _evaluate_scores(scores, target_indices=None):
    top5_count, top1_count = topk(scores, 5, target_indices=target_indices)
    total = scores.shape[0]
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_cosine(query_features, image_features, target_indices=None):
    scores = _cosine_scores(query_features, image_features)
    return _evaluate_scores(scores, target_indices=target_indices)


def _evaluate_fresh_saw_only(query_features, image_features, params):
    transformed, _ = _fit_saw_transform(
        query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
    return _evaluate_cosine(transformed, image_features, target_indices=np.arange(len(transformed), dtype=np.int64))


def _evaluate_fresh_sattc(query_features, image_features, params):
    transformed, _ = _fit_saw_transform(
        query_features,
        shrink=params.saw_shrink,
        diag=params.saw_diag,
        normalize=params.saw_renorm,
    )
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
            scores = _score_query_features(
                transformed,
                image_features,
                use_csls=params.use_csls,
                csls_k=params.csls_k,
            )

    if params.final_sinkhorn_enabled:
        scores = sinkhorn_normalize(scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters)

    return _evaluate_scores(scores, target_indices=np.arange(len(transformed), dtype=np.int64))


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
    for subject_id in subject_ids:
        checkpoint_dir = _find_checkpoint_dir(source_run_dir, subject_id)
        train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
        eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
        eval_args = _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
        test_dataset = _build_test_dataset(eval_args, subject_id)
        modules = _load_modules(eval_args, checkpoint_dir, test_dataset)
        eeg_features, image_features = _encode_full_subject_features(eval_args, modules, test_dataset)
        encoded[int(subject_id)] = {
            "checkpoint_dir": checkpoint_dir,
            "eval_args": eval_args,
            "eeg_features": eeg_features,
            "image_features": image_features,
        }
    return encoded


def _plot_progressive_curves(avg_df, best_df, output_dir, metric):
    if avg_df.empty:
        return

    metric_mean = f"{metric}_mean"
    scopes = sorted(avg_df["scope"].unique().tolist())
    for scope in scopes:
        scope_df = avg_df[avg_df["scope"] == scope].copy()
        if scope_df.empty:
            continue

        plt.figure(figsize=(9, 5.5))
        for alpha in sorted(scope_df["alpha"].unique().tolist()):
            alpha_df = scope_df[scope_df["alpha"] == alpha].sort_values("fit_size")
            plt.plot(alpha_df["fit_size"], alpha_df[metric_mean], marker="o", linewidth=1.5, label=f"alpha={alpha:g}")

        scope_best_df = best_df[best_df["scope"] == scope].sort_values("fit_size")
        if not scope_best_df.empty:
            plt.plot(
                scope_best_df["fit_size"],
                scope_best_df[metric_mean],
                color="black",
                linewidth=2.5,
                linestyle="--",
                label="best alpha",
            )

        plt.xlabel("Calibration fit size")
        plt.ylabel(f"{metric.upper()} accuracy (%)")
        plt.title(f"Progressive frozen-transfer curve: {scope}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{scope}_{metric}.png"), dpi=220)
        plt.close()


def _plot_fit100_alpha(avg_df, output_dir, metric, fit_size=100):
    metric_mean = f"{metric}_mean"
    fit_df = avg_df[avg_df["fit_size"] == fit_size].copy()
    if fit_df.empty:
        return

    plt.figure(figsize=(7.5, 5))
    for scope in sorted(fit_df["scope"].unique().tolist()):
        scope_df = fit_df[fit_df["scope"] == scope].sort_values("alpha")
        plt.plot(scope_df["alpha"], scope_df[metric_mean], marker="o", linewidth=2.0, label=scope)
    plt.xlabel("Alpha blend")
    plt.ylabel(f"{metric.upper()} accuracy (%)")
    plt.title(f"Frozen-transfer alpha sweep at fit size = {fit_size}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fit_{fit_size}_alpha_sweep_{metric}.png"), dpi=220)
    plt.close()


def _build_overview_summary(baseline_avg_df, best_df):
    rows = []
    for _, row in baseline_avg_df.iterrows():
        rows.append(
            {
                "experiment": "baseline",
                "scope": row["scope"],
                "fit_size": np.nan,
                "alpha": np.nan,
                "top1_mean": row["top1_mean"],
                "top5_mean": row["top5_mean"],
            }
        )

    for scope in sorted(best_df["scope"].unique().tolist()):
        scope_df = best_df[best_df["scope"] == scope]
        for fit_size in [25, 50, 100, 150, 195, 200]:
            fit_df = scope_df[scope_df["fit_size"] == fit_size]
            if fit_df.empty:
                continue
            row = fit_df.iloc[0]
            rows.append(
                {
                    "experiment": "best_frozen_transfer",
                    "scope": scope,
                    "fit_size": int(row["fit_size"]),
                    "alpha": float(row["alpha"]),
                    "top1_mean": float(row["top1_mean"]),
                    "top5_mean": float(row["top5_mean"]),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone SATTC transfer experiments that probe how well a learned "
            "orthogonal map transfers after being fit on only a calibration subset."
        )
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300",
        help="Directory containing the per-subject checkpoints for one trained run.",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--progressive_repeats", type=int, default=10)
    parser.add_argument(
        "--fit_sizes",
        nargs="+",
        type=int,
        default=list(range(5, 201, 5)),
        help="Calibration subset sizes to fit the frozen transform on.",
    )
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Blend between identity (0) and learned orthogonal map (1).",
    )
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.94)
    parser.add_argument("--sattc_csls_k", type=int, default=3)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=12)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=16)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.2)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: '{source_run_dir}'")

    if args.output_dir is None:
        run_tag = f"inductive_tta_transfer_suite_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = os.path.join(REPO_ROOT, "results", "things_eeg", "inter-subjects", run_tag)
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)

    fit_sizes = sorted({int(size) for size in args.fit_sizes if int(size) > 0})
    alpha_values = sorted({float(alpha) for alpha in args.alpha_values})

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
        final_sinkhorn_enabled=True,
    )

    print("Encoding all held-out subjects once...")
    encoded = _encode_all_subjects(source_run_dir, args.subjects, args)

    baseline_rows = []
    progressive_rows = []

    # Check for existing results to resume
    baseline_csv = os.path.join(output_dir, "baseline_subject_results.csv")
    progressive_csv = os.path.join(output_dir, "progressive_subject_results.csv")
    if os.path.exists(baseline_csv):
        print(f"Loading existing baselines from {baseline_csv}")
        baseline_rows = pd.read_csv(baseline_csv).to_dict("records")
    if os.path.exists(progressive_csv):
        print(f"Loading existing progressive results from {progressive_csv}")
        progressive_rows = pd.read_csv(progressive_csv).to_dict("records")

    for subject_id in args.subjects:
        subject_id = int(subject_id)
        
        # Skip if subject already fully processed in progressive_rows
        if progressive_rows:
            done_subjects = {int(r["subject_id"]) for r in progressive_rows}
            # We only skip if the subject is completely done (all repeats, fit_sizes, alphas)
            # For simplicity, we check if the subject_id is present at all.
            # If a subject was partially processed, this might skip the rest of it.
            # But usually it fails at the end of a subject or start of a new one.
            if subject_id in done_subjects:
                print(f"Skipping subject {subject_id:02d} (already in {progressive_csv})")
                continue

        subject_data = encoded[subject_id]
        eeg_features = subject_data["eeg_features"]
        image_features = subject_data["image_features"]
        total = eeg_features.shape[0]
        target_indices = np.arange(total, dtype=np.int64)

        print(f"[subject {subject_id:02d}] baselines and progressive transfer")

        plain_metrics = _evaluate_cosine(eeg_features, image_features, target_indices=target_indices)
        saw_metrics = _evaluate_fresh_saw_only(eeg_features, image_features, params)
        full_metrics = _evaluate_fresh_sattc(eeg_features, image_features, params)

        baseline_rows.extend(
            [
                {"subject_id": subject_id, "scope": "full_200", "regime": "plain_cosine", **plain_metrics},
                {"subject_id": subject_id, "scope": "full_200", "regime": "fresh_saw_only", **saw_metrics},
                {"subject_id": subject_id, "scope": "full_200", "regime": "fresh_full_sattc", **full_metrics},
            ]
        )

        clipped_fit_sizes = [size for size in fit_sizes if size <= total]
        rng = np.random.default_rng(args.seed + subject_id)
        for repeat_idx in range(args.progressive_repeats):
            perm = rng.permutation(total)
            for fit_size in clipped_fit_sizes:
                fit_idx = np.sort(perm[:fit_size])
                holdout_idx = np.sort(perm[fit_size:])

                calibration = _fit_frozen_calibration(eeg_features[fit_idx], image_features[fit_idx], params)

                plain_holdout = None
                if holdout_idx.size > 0:
                    plain_holdout = _evaluate_cosine(
                        eeg_features[holdout_idx],
                        image_features,
                        target_indices=holdout_idx,
                    )

                for alpha in alpha_values:
                    transformed_full = _apply_frozen_calibration(eeg_features, calibration, alpha)
                    full_eval_metrics = _evaluate_cosine(
                        transformed_full,
                        image_features,
                        target_indices=target_indices,
                    )
                    progressive_rows.append(
                        {
                            "subject_id": subject_id,
                            "repeat": repeat_idx,
                            "scope": "full_200",
                            "fit_size": fit_size,
                            "alpha": alpha,
                            "top1_acc": round(float(full_eval_metrics["top1_acc"]), 4),
                            "top5_acc": round(float(full_eval_metrics["top5_acc"]), 4),
                            "plain_top1_acc": round(float(plain_metrics["top1_acc"]), 4),
                            "plain_top5_acc": round(float(plain_metrics["top5_acc"]), 4),
                        }
                    )

                    if holdout_idx.size == 0:
                        continue

                    holdout_eval_metrics = _evaluate_cosine(
                        transformed_full[holdout_idx],
                        image_features,
                        target_indices=holdout_idx,
                    )
                    progressive_rows.append(
                        {
                            "subject_id": subject_id,
                            "repeat": repeat_idx,
                            "scope": "holdout_queries_full200_candidates",
                            "fit_size": fit_size,
                            "alpha": alpha,
                            "top1_acc": round(float(holdout_eval_metrics["top1_acc"]), 4),
                            "top5_acc": round(float(holdout_eval_metrics["top5_acc"]), 4),
                            "plain_top1_acc": round(float(plain_holdout["top1_acc"]), 4),
                            "plain_top5_acc": round(float(plain_holdout["top5_acc"]), 4),
                        }
                    )
        
        # Save intermediate results after each subject
        pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
        pd.DataFrame(progressive_rows).to_csv(progressive_csv, index=False)

    baseline_df = pd.DataFrame(baseline_rows)
    # No need to save again here as we save in the loop, but kept for consistency
    baseline_df.to_csv(baseline_csv, index=False)
    baseline_avg_df = (
        baseline_df.groupby(["scope", "regime"], as_index=False)[["top1_acc", "top5_acc"]]
        .mean()
        .rename(columns={"top1_acc": "top1_mean", "top5_acc": "top5_mean"})
    )
    baseline_avg_df.to_csv(os.path.join(output_dir, "baseline_average_results.csv"), index=False)

    progressive_df = pd.DataFrame(progressive_rows)
    progressive_df.to_csv(os.path.join(output_dir, "progressive_subject_results.csv"), index=False)

    progressive_avg_df = (
        progressive_df.groupby(["scope", "fit_size", "alpha"], as_index=False)[
            ["top1_acc", "top5_acc", "plain_top1_acc", "plain_top5_acc"]
        ]
        .agg(["mean", "std"])
    )
    # Flatten the MultiIndex columns created by .agg(["mean", "std"])
    progressive_avg_df.columns = [
        f"{col}_{stat}" if stat else col
        for col, stat in progressive_avg_df.columns
    ]
    progressive_avg_df = progressive_avg_df.reset_index()

    # Ensure column names match what downstream code expects
    progressive_avg_df = progressive_avg_df.rename(columns={
        "top1_acc_mean": "top1_mean",
        "top1_acc_std": "top1_std",
        "top5_acc_mean": "top5_mean",
        "top5_acc_std": "top5_std",
        "plain_top1_acc_mean": "plain_top1_mean",
        "plain_top1_acc_std": "plain_top1_std",
        "plain_top5_acc_mean": "plain_top5_mean",
        "plain_top5_acc_std": "plain_top5_std",
    })
    progressive_avg_df.to_csv(os.path.join(output_dir, "progressive_average_results.csv"), index=False)

    progressive_best_df = (
        progressive_avg_df.sort_values(["scope", "fit_size", "top1_mean", "top5_mean"], ascending=[True, True, False, False])
        .groupby(["scope", "fit_size"], as_index=False)
        .first()
    )
    progressive_best_df.to_csv(os.path.join(output_dir, "progressive_best_alpha_results.csv"), index=False)

    fit100_df = progressive_avg_df[progressive_avg_df["fit_size"] == 100].copy()
    fit100_df.to_csv(os.path.join(output_dir, "fit100_alpha_sweep_results.csv"), index=False)

    overview_df = _build_overview_summary(baseline_avg_df, progressive_best_df)
    overview_df.to_csv(os.path.join(output_dir, "overview_summary.csv"), index=False)

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "source_run_dir": source_run_dir,
                "resolved_fit_sizes": fit_sizes,
                "resolved_alpha_values": alpha_values,
                "resolved_subjects": [int(subject_id) for subject_id in args.subjects],
            },
            f,
            indent=4,
        )

    if not args.skip_plots:
        _plot_progressive_curves(progressive_avg_df, progressive_best_df, output_dir, metric="top1")
        _plot_progressive_curves(progressive_avg_df, progressive_best_df, output_dir, metric="top5")
        _plot_fit100_alpha(progressive_avg_df, output_dir, metric="top1", fit_size=100)
        _plot_fit100_alpha(progressive_avg_df, output_dir, metric="top5", fit_size=100)

    print("\n=== Baseline averages ===")
    print(baseline_avg_df.to_string(index=False))
    print("\n=== Best frozen-transfer curves ===")
    print(progressive_best_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
