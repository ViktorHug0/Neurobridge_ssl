"""Select uniform ensembles on source validation subjects and score outer tests.

The report keeps fixed ensemble sizes separate.  This avoids choosing the size
from an outer-test result while still allowing the members within that size to
be selected independently on each fold's source validation subject.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def scores(path: Path) -> np.ndarray:
    data = np.load(path)
    eeg = data["eeg"].astype(np.float64)
    image = data["image"].astype(np.float64)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    return eeg @ image.T


def accuracy(matrix: np.ndarray) -> float:
    return float(np.mean(matrix.argmax(1) == np.arange(len(matrix))) * 100.0)


def nll(matrix: np.ndarray, temperature: float = 0.05) -> float:
    logits = matrix / temperature
    logits -= logits.max(1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(1))
    return float(np.mean(logsumexp - logits[np.arange(len(logits)), np.arange(len(logits))]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="ensemble_experiments/validation/pool.json")
    parser.add_argument("--dump_dir", default="results/things_eeg/honest_ensemble/dumps")
    parser.add_argument("--min_members", type=int, default=2)
    parser.add_argument("--max_members", type=int, default=5)
    parser.add_argument("--outers", type=int, nargs="+", default=range(1, 11))
    args = parser.parse_args()

    arms = list(json.load(open(args.pool)))
    dump_dir = Path(args.dump_dir)
    rows_by_size = {
        size: [] for size in range(args.min_members, args.max_members + 1)
    }
    adaptive_rows = []
    for outer in args.outers:
        available = []
        val_subject = outer % 10 + 1
        matrices = {}
        for arm in arms:
            test_path = dump_dir / f"{arm}-outer{outer:02d}-test{outer:02d}.npz"
            val_path = dump_dir / f"{arm}-outer{outer:02d}-val{val_subject:02d}.npz"
            if test_path.exists() and val_path.exists():
                available.append(arm)
                matrices[arm] = (scores(val_path), scores(test_path))
        if len(available) < args.min_members:
            continue

        candidates = []
        for size in range(args.min_members, min(args.max_members, len(available)) + 1):
            size_candidates = []
            for combo in itertools.combinations(available, size):
                val_matrix = np.mean([matrices[name][0] for name in combo], axis=0)
                candidate = (accuracy(val_matrix), -nll(val_matrix), combo)
                candidates.append(candidate)
                size_candidates.append(candidate)

            val_acc, neg_val_nll, combo = max(size_candidates)
            test_matrix = np.mean([matrices[name][1] for name in combo], axis=0)
            test_acc = accuracy(test_matrix)
            rows_by_size[size].append(
                (outer, val_subject, val_acc, -neg_val_nll, test_acc, combo)
            )
            print(
                f"k={size} outer={outer:02d} val={val_subject:02d} "
                f"val_top1={val_acc:5.1f} test_top1={test_acc:5.1f} "
                f"members={'+'.join(combo)}"
            )

        val_acc, neg_val_nll, combo = max(candidates)
        test_matrix = np.mean([matrices[name][1] for name in combo], axis=0)
        test_acc = accuracy(test_matrix)
        adaptive_rows.append((outer, val_subject, val_acc, -neg_val_nll, test_acc, combo))
        print(
            f"adaptive outer={outer:02d} val={val_subject:02d} val_top1={val_acc:5.1f} "
            f"test_top1={test_acc:5.1f} members={'+'.join(combo)}"
        )

    for size, rows in rows_by_size.items():
        if rows:
            print(
                f"fixed k={size} honest mean over {len(rows)} folds: "
                f"{np.mean([row[4] for row in rows]):.2f}"
            )
    if adaptive_rows:
        print(
            f"adaptive-k honest mean over {len(adaptive_rows)} folds: "
            f"{np.mean([row[4] for row in adaptive_rows]):.2f}"
        )


if __name__ == "__main__":
    main()
