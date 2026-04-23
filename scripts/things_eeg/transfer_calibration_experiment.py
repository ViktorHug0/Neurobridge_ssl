#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
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
from module.util import (
    apply_orthogonal_map,
    fit_soft_assignment_procrustes,
    sinkhorn_normalize,
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


def _make_params_with_overrides(params, **overrides):
    updated = vars(params).copy()
    updated.update(overrides)
    return SimpleNamespace(**updated)


def _evaluate_plain(query_features, image_features):
    scores = _cosine_scores(query_features, image_features)
    top5_count, top1_count = topk(scores, 5)
    total = len(query_features)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_frozen_transfer(query_features, image_features, calibration):
    transformed = _apply_frozen_calibration(query_features, calibration)
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
    return _evaluate_frozen_transfer(query_features, image_features, blended_calibration)


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

    top5_count, top1_count = topk(scores, 5)
    total = len(transformed)
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def _evaluate_fresh_saw_only(query_features, image_features, params):
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
    test_dataset = EEGPreImageDataset(train=False, **common_kwargs)
    return test_dataset


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

    modules = {
        "model": model,
        "eeg_projector": eeg_projector,
        "img_projector": img_projector,
        "architecture": architecture,
        "device": device,
    }
    return modules


def _forward_feature(eval_args, modules, eeg_backbone_batch):
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
    meta_rows = []

    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(modules["device"])
            image_feature_batch = batch[1].to(modules["device"])
            subject_id_batch = batch[3].to(modules["device"])
            object_idx_batch = batch[4]
            image_idx_batch = batch[5]

            eeg_backbone_batch = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_id_batch)
            eeg_feature_batch = _forward_feature(eval_args, modules, eeg_backbone_batch)
            image_feature_proj = modules["img_projector"](image_feature_batch)

            eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
            image_feature_list.append(image_feature_proj.cpu().numpy())

            for object_idx, image_idx in zip(object_idx_batch.tolist(), image_idx_batch.tolist()):
                meta_rows.append({
                    "object_idx": int(object_idx),
                    "image_idx": int(image_idx),
                })

    return (
        np.concatenate(eeg_feature_list, axis=0),
        np.concatenate(image_feature_list, axis=0),
        meta_rows,
    )


def _write_subject_samples(path, subject_id, subset_indices, metadata_rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "dataset_index", "object_idx", "image_idx"])
        writer.writeheader()
        for dataset_index, row in zip(subset_indices, metadata_rows):
            writer.writerow({
                "subject_id": int(subject_id),
                "dataset_index": int(dataset_index),
                "object_idx": int(row["object_idx"]),
                "image_idx": int(row["image_idx"]),
            })


def _run_subject_experiment(source_run_dir, subject_id, runtime_args, output_dir):
    checkpoint_dir = _find_checkpoint_dir(source_run_dir, subject_id)
    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = _build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)

    test_dataset = _build_test_dataset(eval_args, subject_id)
    expected_test_size = runtime_args.calibration_size + runtime_args.eval_size
    if len(test_dataset) != expected_test_size:
        raise ValueError(
            f"Expected test fold to have exactly {expected_test_size} samples for subject {subject_id}, "
            f"got {len(test_dataset)}."
        )

    modules = _load_modules(eval_args, checkpoint_dir, test_dataset)

    all_test_indices = np.arange(len(test_dataset), dtype=np.int64)
    rng = np.random.default_rng(runtime_args.seed + int(subject_id))
    split_indices = rng.permutation(all_test_indices)
    calibration_indices = sorted(split_indices[:runtime_args.calibration_size].tolist())
    eval_indices = sorted(split_indices[runtime_args.calibration_size:expected_test_size].tolist())

    calibration_query, calibration_image, calibration_meta = _encode_subset_features(
        eval_args, modules, test_dataset, calibration_indices
    )
    eval_query, eval_image, eval_meta = _encode_subset_features(
        eval_args, modules, test_dataset, eval_indices
    )

    params = SimpleNamespace(
        saw_shrink=runtime_args.sattc_saw_shrink,
        saw_diag=runtime_args.sattc_saw_diag,
        saw_renorm=not runtime_args.sattc_saw_no_renorm,
        use_csls=True,
        csls_k=runtime_args.sattc_csls_k,
        soft_procrustes_enabled=True,
        soft_procrustes_steps=runtime_args.sattc_soft_procrustes_steps,
        soft_procrustes_power=runtime_args.sattc_soft_procrustes_power,
        soft_procrustes_normalize_inputs=runtime_args.sattc_soft_procrustes_normalize_inputs,
        sinkhorn_tau=runtime_args.sattc_sinkhorn_tau,
        sinkhorn_iters=runtime_args.sattc_sinkhorn_iters,
        final_sinkhorn_enabled=True,
    )

    frozen_calibration = _fit_frozen_calibration(calibration_query, calibration_image, params)
    frozen_sweep_rows = []
    for frozen_fit_tau in runtime_args.frozen_tau_values:
        tau_params = _make_params_with_overrides(params, sinkhorn_tau=float(frozen_fit_tau))
        tau_calibration = _fit_frozen_calibration(calibration_query, calibration_image, tau_params)
        for frozen_alpha in runtime_args.frozen_alpha_values:
            frozen_alpha = float(frozen_alpha)
            frozen_sweep_rows.extend([
                {
                    "batch": "test_calibration_split",
                    "regime": "frozen_transfer_sweep",
                    "metrics": _evaluate_frozen_transfer_with_alpha(
                        calibration_query, calibration_image, tau_calibration, frozen_alpha
                    ),
                    "frozen_alpha": frozen_alpha,
                    "frozen_fit_tau": float(frozen_fit_tau),
                },
                {
                    "batch": "test_eval_split",
                    "regime": "frozen_transfer_sweep",
                    "metrics": _evaluate_frozen_transfer_with_alpha(
                        eval_query, eval_image, tau_calibration, frozen_alpha
                    ),
                    "frozen_alpha": frozen_alpha,
                    "frozen_fit_tau": float(frozen_fit_tau),
                },
            ])

    result_rows = []
    subject_dir = os.path.join(output_dir, f"sub-{int(subject_id):02d}")
    os.makedirs(subject_dir, exist_ok=True)

    _write_subject_samples(
        os.path.join(subject_dir, "calibration_test_samples.csv"),
        subject_id,
        calibration_indices,
        calibration_meta,
    )
    _write_subject_samples(
        os.path.join(subject_dir, "evaluation_test_samples.csv"),
        subject_id,
        eval_indices,
        eval_meta,
    )

    regimes = [
        ("test_calibration_split", "plain_cosine", _evaluate_plain(calibration_query, calibration_image), None, None),
        ("test_calibration_split", "fresh_saw_only", _evaluate_fresh_saw_only(calibration_query, calibration_image, params), None, None),
        (
            "test_calibration_split",
            "frozen_transform_fit_on_calibration",
            _evaluate_frozen_transfer(calibration_query, calibration_image, frozen_calibration),
            1.0,
            float(params.sinkhorn_tau),
        ),
        ("test_calibration_split", "fresh_full_sattc", _evaluate_fresh_sattc(calibration_query, calibration_image, params), None, None),
        ("test_eval_split", "plain_cosine", _evaluate_plain(eval_query, eval_image), None, None),
        ("test_eval_split", "fresh_saw_only", _evaluate_fresh_saw_only(eval_query, eval_image, params), None, None),
        (
            "test_eval_split",
            "frozen_transform_from_calibration",
            _evaluate_frozen_transfer(eval_query, eval_image, frozen_calibration),
            1.0,
            float(params.sinkhorn_tau),
        ),
        ("test_eval_split", "fresh_full_sattc", _evaluate_fresh_sattc(eval_query, eval_image, params), None, None),
    ]

    for batch_name, regime_name, metrics, frozen_alpha, frozen_fit_tau in regimes:
        result_rows.append({
            "subject_id": int(subject_id),
            "checkpoint_dir": checkpoint_dir,
            "batch": batch_name,
            "regime": regime_name,
            "frozen_alpha": frozen_alpha,
            "frozen_fit_tau": frozen_fit_tau,
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
        })
    for row in frozen_sweep_rows:
        result_rows.append({
            "subject_id": int(subject_id),
            "checkpoint_dir": checkpoint_dir,
            "batch": row["batch"],
            "regime": row["regime"],
            "frozen_alpha": row["frozen_alpha"],
            "frozen_fit_tau": row["frozen_fit_tau"],
            "top1_acc": round(float(row["metrics"]["top1_acc"]), 4),
            "top5_acc": round(float(row["metrics"]["top5_acc"]), 4),
        })

    pd.DataFrame(result_rows).to_csv(os.path.join(subject_dir, "results.csv"), index=False)
    with open(os.path.join(subject_dir, "runtime_config.json"), "w") as f:
        json.dump(
            {
                "subject_id": int(subject_id),
                "checkpoint_dir": checkpoint_dir,
                "calibration_size": runtime_args.calibration_size,
                "eval_size": runtime_args.eval_size,
                "sample_seed": int(runtime_args.seed + int(subject_id)),
                "sattc_params": {
                    "saw_shrink": runtime_args.sattc_saw_shrink,
                    "saw_diag": runtime_args.sattc_saw_diag,
                    "saw_no_renorm": runtime_args.sattc_saw_no_renorm,
                    "csls_k": runtime_args.sattc_csls_k,
                    "sinkhorn_tau": runtime_args.sattc_sinkhorn_tau,
                    "sinkhorn_iters": runtime_args.sattc_sinkhorn_iters,
                    "soft_procrustes_steps": runtime_args.sattc_soft_procrustes_steps,
                    "soft_procrustes_power": runtime_args.sattc_soft_procrustes_power,
                    "soft_procrustes_normalize_inputs": runtime_args.sattc_soft_procrustes_normalize_inputs,
                    "frozen_alpha_values": runtime_args.frozen_alpha_values,
                    "frozen_tau_values": runtime_args.frozen_tau_values,
                },
                "regime_notes": {
                    "plain_cosine": "No batch-specific adaptation.",
                    "fresh_saw_only": "Apply only subject-adaptive whitening (SAW) on the target batch, then score with plain cosine.",
                    "frozen_transform_fit_on_calibration": "Fit SAW + cumulative soft-Procrustes map on the 100-sample calibration split, then apply it back to that same split.",
                    "frozen_transform_from_calibration": "Fit SAW + cumulative soft-Procrustes map on the 100-sample calibration split, then apply it unchanged to the disjoint 100-sample evaluation split.",
                    "frozen_transfer_sweep": "Refit the calibration map with a chosen Sinkhorn temperature, then blend the learned orthogonal map with identity using alpha before applying it.",
                    "fresh_full_sattc": "Recompute SAW + CSLS + soft-Procrustes + final Sinkhorn on the target batch itself.",
                },
            },
            f,
            indent=4,
        )

    return result_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split each 200-sample test fold into 100 calibration samples and 100 evaluation "
            "samples, then test whether SAATC calibration transfers across the split."
        )
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128",
        help="Directory containing per-subject checkpoint folders from the best inter-subject run.",
    )
    parser.add_argument("--held_out_subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2099)
    parser.add_argument("--calibration_size", type=int, default=100)
    parser.add_argument("--eval_size", type=int, default=100)
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.85)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_csls_k", type=int, default=1)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=30)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=10)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.0)
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    parser.add_argument("--frozen_alpha_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--frozen_tau_values", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: '{source_run_dir}'")

    if args.output_dir is None:
        run_tag = f"test_split_transfer_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = os.path.join(REPO_ROOT, "results", "things_eeg", "inter-subjects", run_tag)
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)

    all_rows = []
    for subject_id in args.held_out_subjects:
        print(f"[test_split_transfer] subject={subject_id:02d}")
        subject_rows = _run_subject_experiment(source_run_dir, subject_id, args, output_dir)
        all_rows.extend(subject_rows)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(output_dir, "all_subject_results.csv"), index=False)

    average_df = (
        all_df.groupby(["batch", "regime", "frozen_alpha", "frozen_fit_tau"], as_index=False, dropna=False)[["top1_acc", "top5_acc"]]
        .mean()
        .sort_values(["batch", "regime", "frozen_fit_tau", "frozen_alpha"], na_position="first")
    )
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("\n=== Average Results ===")
    print(average_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
