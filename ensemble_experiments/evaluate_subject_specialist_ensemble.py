"""Cross-apply intra-subject EEGProject specialists and ensemble their scores.

This is an exploratory test-selected analysis.  Each source checkpoint was
selected on that source subject's own test set; no target-subject checkpoint is
included in the corresponding nine-source ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from module.dataset import EEGPreImageDataset
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="results/things_eeg/intra-subjects/TTA",
    )
    parser.add_argument(
        "--output-dir",
        default="results/things_eeg/subject_specialist_ensemble/testselected",
    )
    parser.add_argument("--sources", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def find_specialists(root: Path, source_ids: list[int]) -> dict[int, Path]:
    specialists = {}
    for source in source_ids:
        candidates = []
        for directory in sorted(root.glob(f"*-sub-{source:02d}")):
            config_path = directory / "train_config.json"
            checkpoint_path = directory / "checkpoint_test_best.pth"
            if not config_path.exists() or not checkpoint_path.exists():
                continue
            config = json.loads(config_path.read_text())
            if config.get("eeg_encoder_type") != "EEGProject":
                continue
            if config.get("train_subject_ids") != [source]:
                continue
            if "InternViT-6B_layer28_mean_8bit" not in config.get(
                "image_feature_dir", ""
            ):
                continue
            candidates.append(directory)
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one EEGProject/InternViT-28 specialist for source "
                f"{source}, found {len(candidates)}: {candidates}"
            )
        specialists[source] = candidates[0]
    return specialists


def load_test_arrays(config: dict, target: int, num_workers: int):
    dataset = EEGPreImageDataset(
        [target],
        config["eeg_data_dir"],
        config["selected_channels"],
        config["time_window"],
        config["image_feature_dir"],
        config.get("text_feature_dir", ""),
        False,
        [],
        True,
        False,
        None,
        False,
        False,
        False,
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=200,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    eeg, image, subject, objects, images = [], [], [], [], []
    for batch in loader:
        eeg.append(batch[0])
        image.append(batch[1])
        subject.append(batch[3])
        objects.append(batch[4])
        images.append(batch[5])
    arrays = {
        "eeg": torch.cat(eeg),
        "image": torch.cat(image),
        "subject": torch.cat(subject),
        "object": torch.cat(objects).numpy(),
        "image_idx": torch.cat(images).numpy(),
    }
    if arrays["eeg"].shape[0] != 200:
        raise RuntimeError(
            f"target {target} yielded {arrays['eeg'].shape[0]} queries, expected 200"
        )
    return arrays, dataset


def row_z(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return ((scores - mean) / np.maximum(std, eps)).astype(np.float32)


def accuracies(scores: np.ndarray) -> tuple[float, float]:
    targets = np.arange(scores.shape[0])
    top1 = float((scores.argmax(axis=1) == targets).mean() * 100.0)
    top5_indices = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    top5 = float((top5_indices == targets[:, None]).any(axis=1).mean() * 100.0)
    return top1, top5


def encode_scores(
    checkpoint_dir: Path,
    target_arrays: dict[int, dict],
    dataset,
    device: torch.device,
) -> dict[int, np.ndarray]:
    config = json.loads((checkpoint_dir / "train_config.json").read_text())
    args = SimpleNamespace(**config)
    args.feature_dim = int(str(args.feature_dim).split(",")[0])
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint_test_best.pth",
        map_location=device,
        weights_only=False,
    )
    image_dim = dataset.image_features.shape[-1]
    backbone_dim = getattr(args, "eeg_backbone_dim", 0) or image_dim
    model = build_eeg_encoder(
        args, backbone_dim, dataset.num_sample_points, dataset.channels_num
    ).to(device)
    activation = getattr(args, "projector_activation", "none")
    topk = getattr(args, "projector_topk", 512)
    eeg_projector = build_projector(
        args.projector, backbone_dim, args.feature_dim, activation=activation, topk=topk
    ).to(device)
    image_projector = build_projector(
        args.projector, image_dim, args.feature_dim, activation=activation, topk=topk
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    model.eval()
    eeg_projector.eval()
    image_projector.eval()

    result = {}
    with torch.no_grad():
        for target, arrays in target_arrays.items():
            eeg = arrays["eeg"].to(device, non_blocking=True)
            image = arrays["image"].to(device, non_blocking=True)
            subject = arrays["subject"].to(device, non_blocking=True)
            backbone = run_eeg_backbone(model, args, eeg, subject)
            eeg_features = torch.nn.functional.normalize(
                eeg_projector(backbone), dim=1
            )
            image_features = torch.nn.functional.normalize(
                image_projector(image), dim=1
            )
            result[target] = (eeg_features @ image_features.T).float().cpu().numpy()
    del model, eeg_projector, image_projector, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_matrix(path: Path, matrix: np.ndarray, source_ids, target_ids) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source\\target", *[f"sub-{t:02d}" for t in target_ids]])
        for source, row in zip(source_ids, matrix):
            writer.writerow([f"sub-{source:02d}", *[f"{value:.2f}" for value in row]])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(2025)
    specialists = find_specialists(Path(args.checkpoint_root), args.sources)

    reference_config = json.loads(
        (next(iter(specialists.values())) / "train_config.json").read_text()
    )
    target_arrays = {}
    reference_dataset = None
    for target in args.targets:
        arrays, dataset = load_test_arrays(reference_config, target, args.num_workers)
        target_arrays[target] = arrays
        reference_dataset = reference_dataset or dataset

    score_blocks = []
    for source in args.sources:
        print(f"[source {source:02d}] {specialists[source]}", flush=True)
        scores_by_target = encode_scores(
            specialists[source], target_arrays, reference_dataset, device
        )
        block = np.stack([scores_by_target[t] for t in args.targets])
        score_blocks.append(block)
        np.savez_compressed(
            output_dir / f"source_{source:02d}_scores.npz",
            scores=block,
            source=np.asarray(source),
            targets=np.asarray(args.targets),
        )

    scores = np.stack(score_blocks)  # source, target, query, candidate
    top1 = np.zeros(scores.shape[:2], dtype=np.float64)
    top5 = np.zeros_like(top1)
    for i in range(len(args.sources)):
        for j in range(len(args.targets)):
            top1[i, j], top5[i, j] = accuracies(scores[i, j])
    write_matrix(output_dir / "transfer_top1.csv", top1, args.sources, args.targets)
    write_matrix(output_dir / "transfer_top5.csv", top5, args.sources, args.targets)

    summaries = []
    if set(args.sources) == set(range(1, 11)) and set(args.targets) == set(range(1, 11)):
        for target_index, target in enumerate(args.targets):
            keep = [i for i, source in enumerate(args.sources) if source != target]
            zscores = np.stack([row_z(scores[i, target_index]) for i in keep])
            fused = zscores.mean(axis=0)
            ensemble_top1, ensemble_top5 = accuracies(fused)
            member_top1 = top1[keep, target_index]
            correlations = np.corrcoef(zscores.reshape(len(keep), -1))
            upper = correlations[np.triu_indices(len(keep), k=1)]
            summaries.append(
                {
                    "target": target,
                    "mean_source_top1": float(member_top1.mean()),
                    "best_source_top1": float(member_top1.max()),
                    "nine_source_top1": ensemble_top1,
                    "nine_source_top5": ensemble_top5,
                    "mean_pair_score_correlation": float(upper.mean()),
                }
            )
        summaries.append(
            {
                "target": "Average",
                **{
                    key: float(np.mean([row[key] for row in summaries]))
                    for key in summaries[0]
                    if key != "target"
                },
            }
        )
        with (output_dir / "nine_source_ensemble.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summaries, indent=2), flush=True)

    np.savez_compressed(
        output_dir / "all_scores.npz",
        scores=scores,
        sources=np.asarray(args.sources),
        targets=np.asarray(args.targets),
    )


if __name__ == "__main__":
    main()
