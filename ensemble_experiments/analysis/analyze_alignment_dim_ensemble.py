"""Score the predeclared frozen-encoder alignment-dimension ensemble."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


DEFAULT_DIMS = (32, 64, 128, 256, 512, 1024)
SUBJECTS = tuple(range(1, 11))
TRUTH = np.arange(200)


def row_z(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=-1, keepdims=True)) / np.maximum(
        scores.std(axis=-1, keepdims=True), 1e-6
    )


def load_arm(root: Path, arm: str) -> tuple[np.ndarray, list[np.ndarray]]:
    matrices = []
    labels = []
    for subject in SUBJECTS:
        data = np.load(root / f"{arm}-sub{subject:02d}.npz")
        eeg = data["eeg"].astype(np.float32)
        image = data["image"].astype(np.float32)
        eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
        image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
        matrices.append(row_z(eeg @ image.T).astype(np.float32))
        labels.append(np.stack([data["object"], data["image_idx"]], axis=1))
    return np.stack(matrices), labels


def fold_accuracy(scores: np.ndarray) -> np.ndarray:
    return (scores.argmax(axis=-1) == TRUTH).mean(axis=-1) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("results/things_eeg/synthetic_subjects/ensemble_screen/dumps"),
    )
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ensemble_experiments/analysis/frozen_alignment_dim_ensemble.json"),
    )
    args = parser.parse_args()

    arms = [f"dimfrozen_fd{dim}" for dim in args.dims]
    matrices = []
    reference_labels = None
    for arm in arms:
        scores, labels = load_arm(args.dump_root, arm)
        if reference_labels is None:
            reference_labels = labels
        elif not all(np.array_equal(first, second) for first, second in zip(reference_labels, labels)):
            raise ValueError(f"query order differs for {arm}")
        matrices.append(scores)
    scores = np.stack(matrices)

    solo = {}
    correct = []
    for index, arm in enumerate(arms):
        folds = fold_accuracy(scores[index])
        solo[arm] = {"mean_top1": float(folds.mean()), "fold_top1": folds.tolist()}
        correct.append(scores[index].argmax(axis=-1) == TRUTH)
    correct_array = np.stack(correct)

    correlation = np.eye(len(arms), dtype=np.float64)
    for first, second in itertools.combinations(range(len(arms)), 2):
        value = np.corrcoef(scores[first].ravel(), scores[second].ravel())[0, 1]
        correlation[first, second] = correlation[second, first] = value

    exhaustive = {}
    for size in range(1, len(arms) + 1):
        best = None
        for combination in itertools.combinations(range(len(arms)), size):
            folds = fold_accuracy(scores[list(combination)].sum(axis=0))
            candidate = (float(folds.mean()), combination, folds)
            if best is None or candidate[0] > best[0]:
                best = candidate
        mean, combination, folds = best
        exhaustive[str(size)] = {
            "mean_top1": mean,
            "members": [arms[index] for index in combination],
            "fold_top1": folds.tolist(),
            "combinations_evaluated": int(len(list(itertools.combinations(arms, size)))),
        }

    fixed_folds = fold_accuracy(scores.sum(axis=0))
    output = {
        "protocol": (
            "same fold-specific seed-3300 TSConv encoder; encoder frozen in eval mode; "
            "fresh EEG/image projectors; 10 epochs; test-loss checkpoint selection; "
            "per-query row-z; uniform score mean"
        ),
        "dimensions": args.dims,
        "arms": arms,
        "solo": solo,
        "score_correlation": {
            "matrix": correlation.tolist(),
            "mean_off_diagonal": float(
                correlation[np.triu_indices(len(arms), k=1)].mean()
            ),
        },
        "fixed_all_six": {
            "mean_top1": float(fixed_folds.mean()),
            "fold_top1": fixed_folds.tolist(),
        },
        "individual_member_oracle_top1": float(correct_array.any(axis=0).mean() * 100.0),
        "posthoc_best_by_k": exhaustive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
