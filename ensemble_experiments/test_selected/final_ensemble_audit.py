"""Audit fixed 2--5 member ensembles from stored score dumps.

The script deliberately separates:
  * fixed best (diagnostic, selected on every requested fold),
  * nested LOFO (historical diagnostic only; forbidden as a final rule), and
  * one global sanity-selected rule applied unchanged to every listed fold.

All fusion transforms are label-free functions of each query's candidate scores.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_POOL = (
    "atm_iv",
    "ge100",
    "tsconv_eva",
    "tsconv_vith",
    "iv33",
    "tm3",
    "tsconv_bigg",
    "atm_vith",
)
@dataclass(frozen=True)
class Candidate:
    method: str
    members: tuple[str, ...]


def load_matrix(path: Path) -> np.ndarray:
    data = np.load(path)
    eeg = data["eeg"].astype(np.float32)
    image = data["image"].astype(np.float32)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    matrix = eeg @ image.T
    if matrix.shape != (200, 200):
        raise ValueError(f"expected 200x200 matrix in {path}, got {matrix.shape}")
    return matrix


def row_z(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(1, keepdims=True)) / np.maximum(
        matrix.std(1, keepdims=True), 1e-6
    )


def row_rank(matrix: np.ndarray) -> np.ndarray:
    order = matrix.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = np.arange(matrix.shape[1], dtype=np.float32)[None, :]
    return ranks / max(matrix.shape[1] - 1, 1)


def top1(matrix: np.ndarray) -> float:
    return float(np.mean(matrix.argmax(1) == np.arange(matrix.shape[0])) * 100.0)


def fuse(
    raw: dict[str, list[np.ndarray]],
    transformed: dict[str, dict[str, list[np.ndarray]]],
    candidate: Candidate,
    fold: int,
) -> np.ndarray:
    if candidate.method in (
        "raw",
        "row_z",
        "row_rank",
        "row_pow05",
        "row_pow125",
        "row_softmax4",
    ):
        source = raw if candidate.method == "raw" else transformed[candidate.method]
        return np.mean([source[name][fold] for name in candidate.members], axis=0)
    if candidate.method == "gap025":
        matrices = np.stack([raw[name][fold] for name in candidate.members])
        top_two = np.partition(matrices, -2, axis=2)[:, :, -2:]
        gaps = np.maximum(top_two[:, :, 1] - top_two[:, :, 0], 1e-8) ** 0.25
        weights = gaps / gaps.sum(0, keepdims=True)
        return np.sum(matrices * weights[:, :, None], axis=0)
    raise ValueError(candidate.method)


def describe(label: str, scores: list[float], subjects: tuple[int, ...]) -> None:
    rendered = ", ".join(f"s{s}={v:.1f}" for s, v in zip(subjects, scores))
    print(f"{label}: mean={np.mean(scores):.2f} [{rendered}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-arms", nargs="*", default=[])
    parser.add_argument("--pool", nargs="*", default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--selection-subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="sanity folds used once to choose one global rule for every subject",
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    args = parser.parse_args()

    subjects = tuple(args.subjects)
    if len(subjects) < 2:
        raise ValueError("at least two subjects are required for non-peeking selection")
    names = tuple(dict.fromkeys(tuple(args.pool or DEFAULT_POOL) + tuple(args.new_arms)))
    root = Path(args.dump_root)
    missing = [
        str(root / f"{name}-sub{subject:02d}.npz")
        for name in names
        for subject in subjects
        if not (root / f"{name}-sub{subject:02d}.npz").exists()
    ]
    if missing:
        raise FileNotFoundError("missing score dumps:\n" + "\n".join(missing))

    raw = {
        name: [load_matrix(root / f"{name}-sub{subject:02d}.npz") for subject in subjects]
        for name in names
    }
    transformed = {
        "row_z": {name: [row_z(matrix) for matrix in raw[name]] for name in names},
        "row_rank": {name: [row_rank(matrix) for matrix in raw[name]] for name in names},
    }
    transformed["row_pow05"] = {
        name: [np.sign(matrix) * np.abs(matrix) ** 0.5 for matrix in transformed["row_z"][name]]
        for name in names
    }
    transformed["row_pow125"] = {
        name: [np.sign(matrix) * np.abs(matrix) ** 1.25 for matrix in transformed["row_z"][name]]
        for name in names
    }
    transformed["row_softmax4"] = {
        name: [
            np.exp(matrix / 4.0 - (matrix / 4.0).max(1, keepdims=True))
            / np.exp(matrix / 4.0 - (matrix / 4.0).max(1, keepdims=True)).sum(
                1, keepdims=True
            )
            for matrix in transformed["row_z"][name]
        ]
        for name in names
    }
    methods = (
        "raw",
        "row_z",
        "row_rank",
        "row_pow05",
        "row_pow125",
        "row_softmax4",
        "gap025",
    )
    candidates = [
        Candidate(method, combo)
        for size in range(args.min_k, min(args.max_k, len(names)) + 1)
        for combo in itertools.combinations(names, size)
        for method in methods
    ]
    accuracies = np.empty((len(candidates), len(subjects)), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        for fold in range(len(subjects)):
            accuracies[index, fold] = top1(fuse(raw, transformed, candidate, fold))

    print(f"pool ({len(names)}): {' + '.join(names)}")
    print(f"evaluated {len(candidates)} member/fusion candidates\n")

    print("fixed diagnostic (selection sees every listed fold):")
    for size in range(args.min_k, min(args.max_k, len(names)) + 1):
        indices = [i for i, c in enumerate(candidates) if len(c.members) == size]
        best = max(indices, key=lambda i: (float(accuracies[i].mean()), -i))
        candidate = candidates[best]
        describe(
            f"  k={size} {candidate.method} {' + '.join(candidate.members)}",
            accuracies[best].tolist(),
            subjects,
        )

    nested_scores: list[float] = []
    nested_picks: list[Candidate] = []
    for held in range(len(subjects)):
        train = [fold for fold in range(len(subjects)) if fold != held]
        best = max(
            range(len(candidates)),
            key=lambda i: (float(accuracies[i, train].mean()), -i),
        )
        nested_scores.append(float(accuracies[best, held]))
        nested_picks.append(candidates[best])
    print("\nnested LOFO diagnostic (FORBIDDEN as final: rule changes by subject):")
    describe("  adaptive k=2..5", nested_scores, subjects)
    for subject, pick in zip(subjects, nested_picks):
        print(f"    s{subject}: {pick.method} {' + '.join(pick.members)}")

    selection_subjects = set(args.selection_subjects)
    gate_indices = [i for i, subject in enumerate(subjects) if subject in selection_subjects]
    untouched_indices = [i for i, subject in enumerate(subjects) if subject not in selection_subjects]
    if gate_indices and untouched_indices:
        best = max(
            range(len(candidates)),
            key=lambda i: (float(accuracies[i, gate_indices].mean()), -i),
        )
        candidate = candidates[best]
        print("\nglobal sanity selection (one rule applied unchanged to all folds):")
        print(f"  pick: {candidate.method} {' + '.join(candidate.members)}")
        describe(
            "  gate folds",
            accuracies[best, gate_indices].tolist(),
            tuple(subjects[i] for i in gate_indices),
        )
        describe(
            "  untouched folds",
            accuracies[best, untouched_indices].tolist(),
            tuple(subjects[i] for i in untouched_indices),
        )
        describe("  all folds", accuracies[best].tolist(), subjects)


if __name__ == "__main__":
    main()
