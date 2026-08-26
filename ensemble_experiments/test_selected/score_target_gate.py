"""Score new visual-target arms against the strongest existing member pool."""

import argparse
import itertools
from pathlib import Path

import numpy as np


OLD_QUARTET = ("atm_iv", "ge100", "tsconv_eva", "tsconv_vith")
BASE_POOL = OLD_QUARTET + ("iv33", "tm3", "tsconv_bigg", "atm_vith")


def load_scores(path: Path) -> np.ndarray:
    data = np.load(path)
    eeg = data["eeg"].astype(np.float64)
    image = data["image"].astype(np.float64)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    return eeg @ image.T


def accuracy(matrix: np.ndarray) -> float:
    return float(np.mean(matrix.argmax(1) == np.arange(len(matrix))) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-arms", nargs="+", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    args = parser.parse_args()

    names = tuple(dict.fromkeys(BASE_POOL + tuple(args.new_arms)))
    root = Path(args.dump_root)
    matrices = {
        name: [load_scores(root / f"{name}-sub{subject:02d}.npz") for subject in args.subjects]
        for name in names
    }

    def combo_scores(combo: tuple[str, ...]) -> list[float]:
        return [
            accuracy(np.mean([matrices[name][fold] for name in combo], axis=0))
            for fold in range(len(args.subjects))
        ]

    old_scores = combo_scores(OLD_QUARTET)
    print(f"old quartet: mean={np.mean(old_scores):.2f} folds={old_scores}")
    for arm in args.new_arms:
        solo = combo_scores((arm,))
        plus = combo_scores(OLD_QUARTET + (arm,))
        replacements = []
        for combo in itertools.combinations(OLD_QUARTET + (arm,), 4):
            scores = combo_scores(combo)
            replacements.append((np.mean(scores), combo, scores))
        best = max(replacements)
        print(f"{arm} solo: mean={np.mean(solo):.2f} folds={solo}")
        print(f"{arm} old4+arm: mean={np.mean(plus):.2f} folds={plus}")
        print(f"{arm} best k4 replacement: mean={best[0]:.2f} members={best[1]} folds={best[2]}")

    print("\nbest fixed combinations in focused pool:")
    for size in range(2, 6):
        rows = []
        for combo in itertools.combinations(names, size):
            scores = combo_scores(combo)
            rows.append((np.mean(scores), combo, scores))
        for mean, combo, scores in sorted(rows, reverse=True)[:5]:
            print(f"k={size} mean={mean:.2f} members={combo} folds={scores}")


if __name__ == "__main__":
    main()
