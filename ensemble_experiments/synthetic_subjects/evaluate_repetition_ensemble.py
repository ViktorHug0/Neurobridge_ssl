"""Strict-inductive ensemble over repeated EEG measurements of one test query.

THINGS-EEG2 training examples are averages of four repetitions, whereas the usual
test path averages all 80 repetitions before the nonlinear EEG encoder.  This
diagnostic partitions the 80 repetitions into smaller groups, averages each group
in signal space, encodes every resulting view, and aggregates views belonging to
the same image.  It never estimates anything across test images or uses labels to
form a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import _to_bool
from module.dataset import EEGPreImageDataset, _eeg_cache_key, _eeg_cache_path
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything


def _normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def _accuracy(scores: np.ndarray) -> tuple[float, float]:
    targets = np.arange(scores.shape[0])
    top1 = np.mean(scores.argmax(axis=1) == targets) * 100.0
    top5_indices = np.argpartition(scores, -5, axis=1)[:, -5:]
    top5 = np.mean(np.any(top5_indices == targets[:, None], axis=1)) * 100.0
    return float(top1), float(top5)


def _aggregate_scores(
    views: np.ndarray,
    image_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return label-free aggregation rules for [partition, group, query, dim]."""
    image_unit = _normalize_rows(image_features)
    flat_views = views.reshape(-1, views.shape[-2], views.shape[-1])
    view_unit = _normalize_rows(flat_views)

    # Averaging unit vectors is exactly an arithmetic mean of per-view cosine
    # scores up to a positive query-specific scalar, so no 200x200xV score cube
    # needs to be materialized.
    cosine_centroid = _normalize_rows(view_unit.mean(axis=0))
    feature_centroid = _normalize_rows(flat_views.mean(axis=0))

    # Robust within-query consensus: views aligned with their own unit-vector
    # centroid receive more weight.  Temperature-free nonnegative similarities
    # keep this diagnostic free of target-selected hyperparameters.
    agreement = np.einsum("vqd,qd->vq", view_unit, cosine_centroid)
    agreement = np.maximum(agreement, 0.0)
    agreement /= np.maximum(agreement.sum(axis=0, keepdims=True), 1e-12)
    consensus = _normalize_rows(np.einsum("vq,vqd->qd", agreement, view_unit))

    return {
        "cosine_mean": cosine_centroid @ image_unit.T,
        "feature_mean": feature_centroid @ image_unit.T,
        "agreement_mean": consensus @ image_unit.T,
    }


def _load_config(checkpoint_dir: Path, device: str, subject: int) -> SimpleNamespace:
    with (checkpoint_dir / "train_config.json").open() as handle:
        cfg = json.load(handle)
    cfg["feature_dim"] = int(str(cfg["feature_dim"]).split(",")[0])
    cfg["device"] = device
    cfg["test_subject_id"] = subject
    return SimpleNamespace(**cfg)


def _build_modules(args: SimpleNamespace, checkpoint: dict, dataset, device):
    image_dim = int(dataset.image_features.shape[-1])
    backbone_dim = getattr(args, "eeg_backbone_dim", 0) or image_dim
    activation = getattr(args, "projector_activation", "none")
    topk = getattr(args, "projector_topk", 512)
    model = build_eeg_encoder(
        args, backbone_dim, dataset.num_sample_points, dataset.channels_num
    ).to(device)
    eeg_projector = build_projector(
        args.projector, backbone_dim, args.feature_dim,
        activation=activation, topk=topk,
    ).to(device)
    image_projector = build_projector(
        args.projector, image_dim, args.feature_dim,
        activation=activation, topk=topk,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    model.eval()
    eeg_projector.eval()
    image_projector.eval()
    return model, eeg_projector, image_projector


def _encode_group_means(
    raw_eeg: np.ndarray,
    group_indices: np.ndarray,
    model,
    eeg_projector,
    args,
    subject: int,
    device,
    batch_size: int,
) -> np.ndarray:
    """Encode [query, group, repetition] means and return [group, query, dim]."""
    num_queries, num_groups, _ = group_indices.shape
    encoded_groups = []
    subject_batch = torch.full(
        (batch_size,), int(subject), dtype=torch.long, device=device
    )
    with torch.inference_mode():
        for group_index in range(num_groups):
            group_features = []
            for start in range(0, num_queries, batch_size):
                stop = min(start + batch_size, num_queries)
                # Slice queries before advanced indexing.  For k=80 this bounds
                # the temporary at batch_size rather than copying the full 1-GB
                # repetition tensor.
                indices = group_indices[start:stop, group_index, :]
                grouped = np.take_along_axis(
                    raw_eeg[start:stop],
                    indices[:, :, None, None],
                    axis=1,
                ).mean(axis=1, dtype=np.float32)
                eeg = torch.from_numpy(grouped).to(device)
                subject_ids = subject_batch[: len(eeg)]
                backbone = run_eeg_backbone(model, args, eeg, subject_ids)
                group_features.append(eeg_projector(backbone).cpu().numpy())
            encoded_groups.append(np.concatenate(group_features, axis=0))
    return np.stack(encoded_groups, axis=0)


def _set_per_query_batch_norm(model) -> None:
    """Use current-query BN moments while leaving dropout and all weights frozen."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            # Training-mode BN still uses batch statistics at momentum zero, but
            # cannot contaminate stored source moments across query order.
            module.momentum = 0.0


def _encode_group_means_per_query_bn(
    raw_eeg: np.ndarray,
    group_indices: np.ndarray,
    model,
    eeg_projector,
    args,
    subject: int,
    device,
) -> np.ndarray:
    """Encode all repetition views of each query as its private BN batch."""
    num_queries, num_groups, _ = group_indices.shape
    output = None
    _set_per_query_batch_norm(model)
    eeg_projector.eval()
    with torch.inference_mode():
        for query in range(num_queries):
            indices = group_indices[query]
            grouped = np.take_along_axis(
                raw_eeg[query][None],
                indices[:, :, None, None],
                axis=1,
            ).mean(axis=1, dtype=np.float32)
            eeg = torch.from_numpy(grouped).to(device)
            subject_ids = torch.full(
                (num_groups,), int(subject), dtype=torch.long, device=device
            )
            backbone = run_eeg_backbone(model, args, eeg, subject_ids)
            features = eeg_projector(backbone).cpu().numpy()
            if output is None:
                output = np.empty(
                    (num_groups, num_queries, features.shape[-1]), dtype=np.float32
                )
            output[:, query] = features
    return output


def _partition_indices(
    num_queries: int,
    num_repetitions: int,
    group_size: int,
    seed: int,
) -> np.ndarray:
    if num_repetitions % group_size:
        raise ValueError(
            f"group_size={group_size} must divide {num_repetitions} repetitions"
        )
    rng = np.random.default_rng(seed)
    permutations = np.stack(
        [rng.permutation(num_repetitions) for _ in range(num_queries)], axis=0
    )
    return permutations.reshape(num_queries, num_repetitions // group_size, group_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--test-subject-id", required=True, type=int)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 80])
    parser.add_argument("--num-partitions", type=int, default=4)
    parser.add_argument("--partition-seed", type=int, default=20260813)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--per-query-bn", action="store_true",
        help="normalize each query's repetition views with private BN moments",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-npz", type=Path)
    cli = parser.parse_args()

    checkpoint_dir = cli.checkpoint_dir.resolve()
    args = _load_config(checkpoint_dir, cli.device, cli.test_subject_id)
    seed_everything(getattr(args, "seed", 0))
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    # The ordinary averaged dataset is small and provides image features plus
    # architecture dimensions.  Repetition data are memory-mapped separately.
    dataset = EEGPreImageDataset(
        [cli.test_subject_id], args.eeg_data_dir, args.selected_channels,
        args.time_window, args.image_feature_dir,
        getattr(args, "text_feature_dir", ""), False,
        getattr(args, "aug_image_feature_dirs", []), True, False, None, False,
        _to_bool(getattr(args, "image_test_aug", False)),
        _to_bool(getattr(args, "eeg_test_aug", False)),
        _to_bool(getattr(args, "frozen_eeg_prior", False)),
        subject_ea_align=_to_bool(getattr(args, "subject_ea_align", False)),
    )
    raw_key = _eeg_cache_key(
        subject_id=cli.test_subject_id, train=False, average=False,
        selected_channels=args.selected_channels, time_window=args.time_window,
    )
    raw_path = Path(_eeg_cache_path(args.eeg_data_dir, raw_key))
    if not raw_path.is_file():
        # Build the processed unaveraged cache once using the canonical loader.
        EEGPreImageDataset(
            [cli.test_subject_id], args.eeg_data_dir, args.selected_channels,
            args.time_window, args.image_feature_dir,
            getattr(args, "text_feature_dir", ""), False,
            getattr(args, "aug_image_feature_dirs", []), False, False, None, False,
        )
    raw_eeg = np.load(raw_path, mmap_mode="r")[:, 0]
    if raw_eeg.ndim != 4:
        raise ValueError(f"Expected [query,repetition,channel,time], got {raw_eeg.shape}")

    checkpoint = torch.load(
        checkpoint_dir / "checkpoint_test_best.pth", map_location=device,
        weights_only=False,
    )
    model, eeg_projector, image_projector = _build_modules(
        args, checkpoint, dataset, device
    )
    with torch.inference_mode():
        images = torch.as_tensor(
            dataset.image_features[:, 0], dtype=torch.float32, device=device
        )
        image_features = image_projector(images).cpu().numpy()

    rows = []
    score_dump = {}
    num_queries, num_repetitions = raw_eeg.shape[:2]
    for group_size in cli.group_sizes:
        # A full 80-repetition mean and the collection of single repetitions are
        # invariant to repartitioning, so avoid redundant inference.
        partitions = 1 if group_size in {1, num_repetitions} else cli.num_partitions
        partition_views = []
        for partition in range(partitions):
            indices = _partition_indices(
                num_queries, num_repetitions, group_size,
                cli.partition_seed + partition,
            )
            partition_views.append(
                _encode_group_means_per_query_bn(
                    raw_eeg, indices, model, eeg_projector, args,
                    cli.test_subject_id, device,
                )
                if cli.per_query_bn
                else _encode_group_means(
                    raw_eeg, indices, model, eeg_projector, args,
                    cli.test_subject_id, device, cli.batch_size,
                )
            )
        views = np.stack(partition_views, axis=0)
        for aggregation, scores in _aggregate_scores(views, image_features).items():
            top1, top5 = _accuracy(scores)
            rows.append({
                "subject": cli.test_subject_id,
                "group_size": group_size,
                "num_partitions": partitions,
                "num_views": int(views.shape[0] * views.shape[1]),
                "per_query_bn": cli.per_query_bn,
                "aggregation": aggregation,
                "top1": top1,
                "top5": top5,
            })
            score_dump[f"k{group_size}_{aggregation}"] = scores.astype(np.float32)

    frame = pd.DataFrame(rows)
    cli.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cli.output_csv, index=False)
    print(frame.to_string(index=False))
    if cli.output_npz is not None:
        cli.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cli.output_npz,
            **score_dump,
            image_features=image_features.astype(np.float32),
        )


if __name__ == "__main__":
    main()
