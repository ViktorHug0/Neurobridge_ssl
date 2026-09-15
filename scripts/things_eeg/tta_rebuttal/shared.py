#!/usr/bin/env python3
"""Shared utilities for rebuttal-focused TTA experiments.

The scripts in this folder intentionally keep the experimental interface small:
load a finished LOSO checkpoint run, encode each held-out subject once, and run
label-free SAW + CSLS + Sinkhorn/Procrustes calibration variants.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize, topk
from train import build_eeg_encoder, build_projector, run_eeg_backbone


DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "results", "things_eeg", "tta_rebuttal")
DEFAULT_SOURCE_RUN_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "things_eeg",
    "inter-subjects",
    "tsconv_dropout_sweep_20260429-190741",
    "param_k30_pool51_do050_featdim512_seed3300",
)


@dataclass
class TTAParams:
    saw_shrink: float = 0.94
    saw_diag: bool = False
    saw_renorm: bool = True
    use_csls: bool = True
    csls_k: int = 3
    sinkhorn_tau: float = 0.1
    sinkhorn_iters: int = 12
    soft_procrustes_steps: int = 16
    soft_procrustes_power: float = 1.2
    soft_procrustes_normalize_inputs: bool = False
    final_sinkhorn: bool = False
    sinkhorn_col_mass: float = 1.0  # 1.0 = bijective prior; R for balanced R-to-1


def add_common_args(parser):
    parser.add_argument("--source_run_dir", type=str, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.94)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_csls_k", type=int, default=3)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=12)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=16)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.2)
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    parser.add_argument("--sattc_final_sinkhorn", action="store_true")
    parser.add_argument("--sattc_sinkhorn_col_mass", type=float, default=1.0)


def params_from_args(args) -> TTAParams:
    return TTAParams(
        saw_shrink=float(args.sattc_saw_shrink),
        saw_diag=bool(args.sattc_saw_diag),
        saw_renorm=not bool(args.sattc_saw_no_renorm),
        use_csls=True,
        csls_k=int(args.sattc_csls_k),
        sinkhorn_tau=float(args.sattc_sinkhorn_tau),
        sinkhorn_iters=int(args.sattc_sinkhorn_iters),
        soft_procrustes_steps=int(args.sattc_soft_procrustes_steps),
        soft_procrustes_power=float(args.sattc_soft_procrustes_power),
        soft_procrustes_normalize_inputs=bool(args.sattc_soft_procrustes_normalize_inputs),
        final_sinkhorn=bool(args.sattc_final_sinkhorn),
        sinkhorn_col_mass=float(getattr(args, "sattc_sinkhorn_col_mass", 1.0)),
    )


def ensure_output_dir(args, experiment_name: str) -> str:
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        from datetime import datetime

        output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"{experiment_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def write_config(output_dir: str, args, params: TTAParams):
    payload = vars(args).copy()
    payload["tta_params"] = asdict(params)
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_rows(features, eps: float = 1e-12):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, eps, None)


def cosine_scores(query_features, image_features):
    return (normalize_rows(query_features) @ normalize_rows(image_features).T).astype(np.float32, copy=False)


def csls_scores(similarities, k: int = 3):
    similarities = np.asarray(similarities, dtype=np.float32)
    n_q, n_c = similarities.shape
    if n_q == 0 or n_c == 0:
        return similarities
    k_eff = max(1, min(int(k), n_q, n_c))
    rx = np.partition(similarities, kth=n_c - k_eff, axis=1)[:, -k_eff:].mean(axis=1, keepdims=True)
    ry = np.partition(similarities, kth=n_q - k_eff, axis=0)[-k_eff:, :].mean(axis=0, keepdims=True)
    return (2.0 * similarities - rx - ry).astype(np.float32, copy=False)


def score_features(query_features, image_features, use_csls: bool = False, csls_k: int = 3):
    scores = cosine_scores(query_features, image_features)
    if use_csls:
        scores = csls_scores(scores, k=csls_k)
    return scores


def evaluate_scores(scores, target_indices=None):
    if target_indices is None:
        target_indices = np.arange(scores.shape[0], dtype=np.int64)
    top5_count, top1_count = topk(scores, 5, target_indices=np.asarray(target_indices, dtype=np.int64))
    total = scores.shape[0]
    return {
        "top1_acc": 100.0 * top1_count / total,
        "top5_acc": 100.0 * top5_count / total,
    }


def fit_saw_transform(features, params: TTAParams, eps: float = 1e-6):
    features = np.asarray(features, dtype=np.float32)
    mu = features.mean(axis=0, keepdims=True)
    centered = features - mu
    cov = centered.T @ centered / float(max(features.shape[0] - 1, 1))
    cov = cov.astype(np.float32, copy=False)
    if params.saw_diag:
        cov = np.diag(np.diag(cov))
    dim = cov.shape[0]
    trace_mean = float(np.trace(cov)) / max(dim, 1)
    cov = (1.0 - params.saw_shrink) * cov + params.saw_shrink * trace_mean * np.eye(dim, dtype=np.float32)
    cov = cov + eps * np.eye(dim, dtype=np.float32)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    whitener = (evecs @ np.diag(evals ** -0.5) @ evecs.T).astype(np.float32, copy=False)
    transformed = (features - mu) @ whitener
    if params.saw_renorm:
        transformed = normalize_rows(transformed)
    return transformed.astype(np.float32, copy=False), {"mu": mu, "whitener": whitener, "renorm": params.saw_renorm}


def apply_saw_transform(features, saw_stats):
    transformed = (np.asarray(features, dtype=np.float32) - saw_stats["mu"]) @ saw_stats["whitener"]
    if saw_stats["renorm"]:
        transformed = normalize_rows(transformed)
    return transformed.astype(np.float32, copy=False)


def project_to_orthogonal(matrix):
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float32), full_matrices=False)
    return (u @ vt).astype(np.float32, copy=False)


def blend_orthogonal_map(orthogonal_map, alpha: float):
    orthogonal_map = np.asarray(orthogonal_map, dtype=np.float32)
    dim = orthogonal_map.shape[0]
    blended = (1.0 - float(alpha)) * np.eye(dim, dtype=np.float32) + float(alpha) * orthogonal_map
    return project_to_orthogonal(blended)


def fit_tta_calibration(query_features, image_features, params: TTAParams):
    transformed, saw_stats = fit_saw_transform(query_features, params)
    cumulative_map = np.eye(transformed.shape[1], dtype=np.float32)
    scores = score_features(transformed, image_features, use_csls=params.use_csls, csls_k=params.csls_k)
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        assignment = sinkhorn_normalize(
            scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters, col_mass=params.sinkhorn_col_mass
        )
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
        scores = score_features(transformed, image_features, use_csls=params.use_csls, csls_k=params.csls_k)
    return {"saw_stats": saw_stats, "orthogonal_map": cumulative_map}


def apply_tta_calibration(query_features, calibration, alpha: float = 1.0):
    transformed = apply_saw_transform(query_features, calibration["saw_stats"])
    transformed = apply_orthogonal_map(transformed, blend_orthogonal_map(calibration["orthogonal_map"], alpha))
    return np.asarray(transformed, dtype=np.float32)


def evaluate_plain(query_features, image_features, target_indices=None):
    return evaluate_scores(cosine_scores(query_features, image_features), target_indices=target_indices)


def evaluate_saw_csls(query_features, image_features, params: TTAParams, target_indices=None):
    transformed, _ = fit_saw_transform(query_features, params)
    scores = score_features(transformed, image_features, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores, target_indices=target_indices)


def evaluate_full_tta(query_features, image_features, params: TTAParams, target_indices=None):
    calibration = fit_tta_calibration(query_features, image_features, params)
    transformed = apply_tta_calibration(query_features, calibration, alpha=1.0)
    scores = score_features(transformed, image_features, use_csls=params.use_csls, csls_k=params.csls_k)
    if params.final_sinkhorn:
        scores = sinkhorn_normalize(scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters)
    return evaluate_scores(scores, target_indices=target_indices)


def find_checkpoint_dir(source_run_dir: str, subject_id: int):
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


def build_eval_args(train_cfg: dict, eval_cfg: dict, runtime_args, subject_id: int):
    merged = {}
    merged.update(train_cfg)
    merged.update(eval_cfg)
    merged["test_subject_id"] = int(subject_id)
    merged["device"] = runtime_args.device
    merged["eval_batch_size"] = runtime_args.batch_size
    merged["num_workers"] = runtime_args.num_workers
    return SimpleNamespace(**merged)


def build_test_dataset(eval_args, subject_id: int, average: bool | None = None):
    if average is None:
        average = to_bool(getattr(eval_args, "data_average", True), True)
    return EEGPreImageDataset(
        subject_ids=[int(subject_id)],
        eeg_data_dir=eval_args.eeg_data_dir,
        selected_channels=eval_args.selected_channels,
        time_window=eval_args.time_window,
        image_feature_dir=eval_args.image_feature_dir,
        text_feature_dir=getattr(eval_args, "text_feature_dir", ""),
        image_aug=False,
        aug_image_feature_dirs=[],
        average=bool(average),
        _random=False,
        eeg_transform=None,
        train=False,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )


def build_train_dataset(eval_args, subject_id: int, average: bool | None = None):
    if average is None:
        average = to_bool(getattr(eval_args, "data_average", True), True)
    return EEGPreImageDataset(
        subject_ids=[int(subject_id)],
        eeg_data_dir=eval_args.eeg_data_dir,
        selected_channels=eval_args.selected_channels,
        time_window=eval_args.time_window,
        image_feature_dir=eval_args.image_feature_dir,
        text_feature_dir=getattr(eval_args, "text_feature_dir", ""),
        image_aug=False,
        aug_image_feature_dirs=[],
        average=bool(average),
        _random=False,
        eeg_transform=None,
        train=True,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )


def load_modules(eval_args, checkpoint_dir: str, test_dataset):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_test_best.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at '{checkpoint_path}'.")
    device = torch.device(eval_args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_feature_dim = test_dataset.image_features.shape[-1]
    backbone_feature_dim = getattr(eval_args, "eeg_backbone_dim", 0) or image_feature_dim
    model = build_eeg_encoder(eval_args, backbone_feature_dim, test_dataset.num_sample_points, test_dataset.channels_num).to(device)
    eeg_projector = build_projector(eval_args.projector, backbone_feature_dim, eval_args.feature_dim).to(device)
    img_projector = build_projector(eval_args.projector, image_feature_dim, eval_args.feature_dim).to(device)
    architecture = getattr(eval_args, "architecture", checkpoint.get("architecture", "baseline"))
    if architecture != "baseline":
        raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    model.eval()
    eeg_projector.eval()
    img_projector.eval()
    return {"model": model, "eeg_projector": eeg_projector, "img_projector": img_projector, "device": device}


def load_subject_context(source_run_dir: str, runtime_args, subject_id: int, average: bool = True):
    checkpoint_dir = find_checkpoint_dir(source_run_dir, subject_id)
    train_cfg = load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
    dataset = build_test_dataset(eval_args, subject_id, average=average)
    modules = load_modules(eval_args, checkpoint_dir, dataset)
    return checkpoint_dir, eval_args, dataset, modules


def load_subject_train_test_context(source_run_dir: str, runtime_args, subject_id: int, average: bool = True):
    checkpoint_dir = find_checkpoint_dir(source_run_dir, subject_id)
    train_cfg = load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = build_eval_args(train_cfg, eval_cfg, runtime_args, subject_id)
    test_dataset = build_test_dataset(eval_args, subject_id, average=average)
    train_dataset = build_train_dataset(eval_args, subject_id, average=average)
    modules = load_modules(eval_args, checkpoint_dir, test_dataset)
    return checkpoint_dir, eval_args, train_dataset, test_dataset, modules


def encode_average_features(eval_args, modules, dataset):
    loader = DataLoader(
        dataset,
        batch_size=getattr(eval_args, "eval_batch_size", 200),
        shuffle=False,
        num_workers=getattr(eval_args, "num_workers", 0),
    )
    eeg_features = []
    image_features = []
    object_indices = []
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(modules["device"])
            image_batch = batch[1].to(modules["device"])
            subject_batch = batch[3].to(modules["device"])
            eeg_backbone = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_batch)
            eeg_features.append(modules["eeg_projector"](eeg_backbone).cpu().numpy())
            image_features.append(modules["img_projector"](image_batch).cpu().numpy())
            object_indices.append(batch[4].numpy())
    order = np.argsort(np.concatenate(object_indices, axis=0))
    return (
        np.concatenate(eeg_features, axis=0)[order].astype(np.float32, copy=False),
        np.concatenate(image_features, axis=0)[order].astype(np.float32, copy=False),
    )


def encode_indexed_average_features(eval_args, modules, dataset):
    """Encode averaged EEG/image pairs, preserving deterministic object-image order."""
    loader = DataLoader(
        dataset,
        batch_size=getattr(eval_args, "eval_batch_size", 200),
        shuffle=False,
        num_workers=getattr(eval_args, "num_workers", 0),
    )
    eeg_features = []
    image_features = []
    object_indices = []
    image_indices = []
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(modules["device"])
            image_batch = batch[1].to(modules["device"])
            subject_batch = batch[3].to(modules["device"])
            eeg_backbone = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_batch)
            eeg_features.append(modules["eeg_projector"](eeg_backbone).cpu().numpy())
            image_features.append(modules["img_projector"](image_batch).cpu().numpy())
            object_indices.append(batch[4].numpy())
            image_indices.append(batch[5].numpy())

    objects = np.concatenate(object_indices, axis=0)
    images = np.concatenate(image_indices, axis=0)
    order = np.lexsort((images, objects))
    return (
        np.concatenate(eeg_features, axis=0)[order].astype(np.float32, copy=False),
        np.concatenate(image_features, axis=0)[order].astype(np.float32, copy=False),
        objects[order].astype(np.int64, copy=False),
        images[order].astype(np.int64, copy=False),
    )


def _test_image_matrix(dataset):
    image_features = np.asarray(dataset.image_features)
    if image_features.ndim == 3:
        image_features = image_features[:, 0, :]
    return image_features.astype(np.float32, copy=False)


def select_repetition_indices(num_repetitions: int, count: int, seed: int, mode: str, num_objects: int):
    count = max(1, min(int(count), int(num_repetitions)))
    mode = str(mode)
    if mode == "first":
        return np.tile(np.arange(count, dtype=np.int64), (num_objects, 1))
    rng = np.random.default_rng(seed)
    if mode == "random_shared":
        reps = np.sort(rng.choice(num_repetitions, size=count, replace=False)).astype(np.int64)
        return np.tile(reps, (num_objects, 1))
    if mode == "random_per_sample":
        return np.stack(
            [np.sort(rng.choice(num_repetitions, size=count, replace=False)).astype(np.int64) for _ in range(num_objects)],
            axis=0,
        )
    raise ValueError(f"Unsupported repetition selection mode: {mode}")


def encode_repetition_limited_features(eval_args, modules, dataset, repetition_count: int, seed: int, selection_mode: str):
    if dataset.average:
        raise ValueError("Repetition-limited encoding requires a dataset built with average=False.")
    eeg_data = np.asarray(dataset.eeg_data_list[0])
    if eeg_data.ndim != 5 or eeg_data.shape[1] != 1:
        raise ValueError(f"Expected test EEG shape (objects, 1, reps, channels, time), got {eeg_data.shape}.")
    num_objects, _, num_repetitions, _, _ = eeg_data.shape
    rep_indices = select_repetition_indices(num_repetitions, repetition_count, seed, selection_mode, num_objects)
    averaged = np.empty((num_objects, eeg_data.shape[-2], eeg_data.shape[-1]), dtype=np.float32)
    for object_idx in range(num_objects):
        averaged[object_idx] = eeg_data[object_idx, 0, rep_indices[object_idx]].mean(axis=0, dtype=np.float32)

    raw_images = _test_image_matrix(dataset)
    eeg_features = []
    image_features = []
    batch_size = int(getattr(eval_args, "eval_batch_size", 200))
    subject_ids = torch.full((num_objects,), int(eval_args.test_subject_id), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, num_objects, batch_size):
            end = min(start + batch_size, num_objects)
            eeg_batch = torch.tensor(averaged[start:end], dtype=torch.float32, device=modules["device"])
            image_batch = torch.tensor(raw_images[start:end], dtype=torch.float32, device=modules["device"])
            subject_batch = subject_ids[start:end].to(modules["device"])
            eeg_backbone = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_batch)
            eeg_features.append(modules["eeg_projector"](eeg_backbone).cpu().numpy())
            image_features.append(modules["img_projector"](image_batch).cpu().numpy())
    return (
        np.concatenate(eeg_features, axis=0).astype(np.float32, copy=False),
        np.concatenate(image_features, axis=0).astype(np.float32, copy=False),
        int(num_repetitions),
    )


def encode_repetition_blocks(eval_args, modules, dataset, block_sizes, seed: int):
    """Encode one query set per disjoint block of EEG repetitions.

    ``block_sizes`` is a list of repetition counts; block b averages its own slice
    of a per-object random permutation of the available repetitions, so the blocks
    are disjoint measurements of the same 200 stimuli. Returns
    (list_of_query_matrices, image_features, num_repetitions).
    """
    if dataset.average:
        raise ValueError("Repetition-block encoding requires a dataset built with average=False.")
    eeg_data = np.asarray(dataset.eeg_data_list[0])
    if eeg_data.ndim != 5 or eeg_data.shape[1] != 1:
        raise ValueError(f"Expected test EEG shape (objects, 1, reps, channels, time), got {eeg_data.shape}.")
    num_objects, _, num_repetitions, _, _ = eeg_data.shape
    block_sizes = [int(b) for b in block_sizes]
    if sum(block_sizes) > num_repetitions:
        raise ValueError(f"Blocks need {sum(block_sizes)} repetitions but only {num_repetitions} available.")

    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(num_repetitions) for _ in range(num_objects)], axis=0)
    raw_images = _test_image_matrix(dataset)
    batch_size = int(getattr(eval_args, "eval_batch_size", 200))
    subject_ids = torch.full((num_objects,), int(eval_args.test_subject_id), dtype=torch.long)

    query_blocks = []
    image_features = None
    offset = 0
    for block_size in block_sizes:
        averaged = np.empty((num_objects, eeg_data.shape[-2], eeg_data.shape[-1]), dtype=np.float32)
        for object_idx in range(num_objects):
            reps = perms[object_idx, offset : offset + block_size]
            averaged[object_idx] = eeg_data[object_idx, 0, reps].mean(axis=0, dtype=np.float32)
        offset += block_size

        eeg_features = []
        images = []
        with torch.no_grad():
            for start in range(0, num_objects, batch_size):
                end = min(start + batch_size, num_objects)
                eeg_batch = torch.tensor(averaged[start:end], dtype=torch.float32, device=modules["device"])
                image_batch = torch.tensor(raw_images[start:end], dtype=torch.float32, device=modules["device"])
                subject_batch = subject_ids[start:end].to(modules["device"])
                eeg_backbone = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subject_batch)
                eeg_features.append(modules["eeg_projector"](eeg_backbone).cpu().numpy())
                images.append(modules["img_projector"](image_batch).cpu().numpy())
        query_blocks.append(np.concatenate(eeg_features, axis=0).astype(np.float32, copy=False))
        if image_features is None:
            image_features = np.concatenate(images, axis=0).astype(np.float32, copy=False)

    return query_blocks, image_features, int(num_repetitions)


def aggregate_results(df: pd.DataFrame, group_cols):
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top5_mean=("top5_acc", "mean"),
            top5_std=("top5_acc", "std"),
        )
        .fillna(0.0)
        .sort_values(group_cols)
    )


# --- Few-shot subject adaptation helpers ---

DEFAULT_FEWSHOT_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "rebuttal_suite_20260602-134420",
    "fewshot_subject_adaptation",
)


def fit_paired_orthogonal(query_features, image_features, eps: float = 1e-8):
    """Paired Procrustes: orthogonal map aligning query rows to paired image rows."""
    query_features = np.asarray(query_features, dtype=np.float32)
    image_features = np.asarray(image_features, dtype=np.float32)
    if query_features.shape[0] == 0:
        dim = query_features.shape[1] if query_features.ndim == 2 else 0
        return np.eye(max(dim, 1), dtype=np.float32)
    cross = query_features.T @ image_features
    try:
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(query_features.shape[1], dtype=np.float32)
    rotation = (u @ vt).astype(np.float32, copy=False)
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = (u @ vt).astype(np.float32, copy=False)
    return rotation


def fit_paired_ridge(query_features, image_features, ridge: float = 1.0):
    """Ridge regression W minimizing ||Q W - I||_F^2 + ridge ||W||_F^2."""
    query_features = np.asarray(query_features, dtype=np.float32)
    image_features = np.asarray(image_features, dtype=np.float32)
    dim = query_features.shape[1]
    q = query_features
    t = image_features
    gram = q.T @ q + float(ridge) * np.eye(dim, dtype=np.float32)
    rhs = q.T @ t
    try:
        weights = np.linalg.solve(gram, rhs).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram, rhs, rcond=None)[0].astype(np.float32, copy=False)
    return weights


def apply_linear_map(features, weights):
    features = np.asarray(features, dtype=np.float32)
    return (features @ np.asarray(weights, dtype=np.float32)).astype(np.float32, copy=False)


def fit_train_zca_stats(query_features, params: TTAParams):
    """Fit SAW/ZCA on train EEG only; return stats for frozen application."""
    _, saw_stats = fit_saw_transform(query_features, params)
    return saw_stats


def apply_train_zca(query_features, saw_stats):
    return apply_saw_transform(query_features, saw_stats)


class LowRankResidualAdapter(torch.nn.Module):
    """ENIGMA-style low-rank residual: x + U(Vx), then L2-normalize."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        rank = max(1, min(int(rank), int(dim)))
        self.u = torch.nn.Linear(rank, dim, bias=False)
        self.v = torch.nn.Linear(dim, rank, bias=False)
        torch.nn.init.normal_(self.u.weight, std=1e-3)
        torch.nn.init.normal_(self.v.weight, std=1e-3)

    def forward(self, x):
        return x + self.u(self.v(x))


class EmbeddingResidualAdapter(torch.nn.Module):
    """Residual MLP adapter: x + scale * MLP(LayerNorm(x)), L2-normalized."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        layers = [torch.nn.LayerNorm(dim)]
        in_dim = dim
        for _ in range(max(1, int(num_layers))):
            layers.extend(
                [
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.residual_scale * self.mlp(x)


class CayleyOrthogonalAdapter(torch.nn.Module):
    """Trainable orthogonal map W = (I - A)(I + A)^-1 with A skew-symmetric."""

    def __init__(self, dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.raw = torch.nn.Parameter(init_scale * torch.randn(dim, dim))

    def orthogonal(self):
        skew = self.raw - self.raw.T
        eye = torch.eye(skew.shape[0], dtype=skew.dtype, device=skew.device)
        return torch.linalg.solve(eye + skew, eye - skew)

    def forward(self, x):
        return x @ self.orthogonal()


@dataclass
class NeuralAdapterConfig:
    hidden_dim: int = 0
    num_layers: int = 1
    dropout: float = 0.1
    residual_scale: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 20
    mse_weight: float = 0.5
    temperature: float = 0.07
    device: str = "cpu"


def _retrieval_metrics_from_transformed(
    transformed_query,
    candidates,
    targets,
    params: TTAParams,
    use_csls: bool = True,
):
    scores = score_features(
        transformed_query,
        candidates,
        use_csls=use_csls and params.use_csls,
        csls_k=params.csls_k,
    )
    return evaluate_scores(scores, target_indices=targets)


def fit_low_rank_adapter(
    train_query,
    train_images,
    val_query,
    val_images,
    all_candidates,
    val_targets,
    rank: int,
    config: NeuralAdapterConfig,
    params: TTAParams,
):
    device = torch.device(config.device)
    dim = train_query.shape[1]
    model = LowRankResidualAdapter(dim, rank).to(device)
    return _train_adapter_module(
        model,
        train_query,
        train_images,
        val_query,
        val_images,
        all_candidates,
        val_targets,
        config,
        params,
        normalize_output=True,
    )


def fit_neural_adapter(
    train_query,
    train_images,
    val_query,
    val_images,
    all_candidates,
    val_targets,
    config: NeuralAdapterConfig,
    params: TTAParams,
):
    device = torch.device(config.device)
    dim = train_query.shape[1]
    hidden = config.hidden_dim if config.hidden_dim > 0 else 2 * dim
    model = EmbeddingResidualAdapter(
        dim,
        hidden,
        num_layers=config.num_layers,
        dropout=config.dropout,
        residual_scale=config.residual_scale,
    ).to(device)
    return _train_adapter_module(
        model,
        train_query,
        train_images,
        val_query,
        val_images,
        all_candidates,
        val_targets,
        config,
        params,
        normalize_output=True,
    )


def _train_adapter_module(
    model,
    train_query,
    train_images,
    val_query,
    val_images,
    all_candidates,
    val_targets,
    config: NeuralAdapterConfig,
    params: TTAParams,
    normalize_output: bool = True,
):
    device = torch.device(config.device)
    model = model.to(device)
    train_q = torch.tensor(train_query, dtype=torch.float32, device=device)
    train_i = torch.tensor(train_images, dtype=torch.float32, device=device)
    val_q = torch.tensor(val_query, dtype=torch.float32, device=device)
    val_i_np = np.asarray(val_images, dtype=np.float32)
    candidates_np = np.asarray(all_candidates, dtype=np.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_val_top1 = -1.0
    stale = 0
    tau = max(float(config.temperature), 1e-6)

    for _epoch in range(int(config.max_epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(train_q)
        if normalize_output:
            out = torch.nn.functional.normalize(out, dim=1)
        logits = (out @ train_i.T) / tau
        labels = torch.arange(train_q.shape[0], device=device)
        loss_ce = torch.nn.functional.cross_entropy(logits, labels)
        loss_mse = torch.nn.functional.mse_loss(out, train_i)
        loss = loss_ce + float(config.mse_weight) * loss_mse
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_q)
            if normalize_output:
                val_out = torch.nn.functional.normalize(val_out, dim=1)
            val_out_np = val_out.cpu().numpy()
        metrics = _retrieval_metrics_from_transformed(val_out_np, candidates_np, val_targets, params)
        if metrics["top1_acc"] > best_val_top1:
            best_val_top1 = metrics["top1_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"module": model, "normalize_output": normalize_output, "best_val_top1": best_val_top1}


def apply_torch_adapter(adapter_bundle, query_features):
    model = adapter_bundle["module"]
    normalize_output = adapter_bundle.get("normalize_output", True)
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(query_features, dtype=torch.float32, device=device)
        out = model(x)
        if normalize_output:
            out = torch.nn.functional.normalize(out, dim=1)
        return out.cpu().numpy().astype(np.float32, copy=False)


def apply_supervised_orthogonal(query_features, rotation, alpha: float = 1.0):
    rotation = blend_orthogonal_map(rotation, alpha)
    return apply_orthogonal_map(query_features, rotation)


def apply_supervised_pipeline(query_features, saw_stats, map_weights, map_kind: str, alpha: float = 1.0):
    """map_kind: 'orthogonal' | 'linear' | 'none'."""
    x = apply_train_zca(query_features, saw_stats) if saw_stats is not None else np.asarray(query_features, dtype=np.float32)
    if map_kind == "orthogonal":
        return apply_supervised_orthogonal(x, map_weights, alpha=alpha)
    if map_kind == "linear":
        return apply_linear_map(x, map_weights)
    return x
