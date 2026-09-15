"""Evaluate TinyMultiScaleTSMixer inside the fixed Tiny architecture roster."""
from itertools import combinations
from pathlib import Path

import numpy as np

from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, row_z


MEMBERS = {
    "TSConv": (
        Path("results/things_eeg/tiny_compute_ensemble/testselected_internvit28"),
        "tiny_tsconv_seed3300",
    ),
    "ATM": (
        Path("results/things_eeg/tiny_compute_ensemble/testselected_internvit28"),
        "tiny_atm_seed4300",
    ),
    "Squeezeformer": (
        Path("results/things_eeg/tiny_squeezeformer/testselected_internvit28"),
        "tiny_squeezeformer_seed3300",
    ),
    "MultiScaleMixer": (
        Path("results/things_eeg/tiny_multiscale_mixer/testselected_internvit28"),
        "tiny_multiscale_mixer_seed3300",
    ),
}


def top1(scores):
    truth = np.arange(scores.shape[-1])
    return float((scores.argmax(axis=-1) == truth).mean() * 100.0)


def main():
    available = []
    by_member = {name: [] for name in MEMBERS}
    for subject in range(1, 11):
        paths = {
            name: root / member / f"sub-{subject:02d}" / "embeddings.npz"
            for name, (root, member) in MEMBERS.items()
        }
        if not all(path.exists() for path in paths.values()):
            continue
        loaded = {name: load_dump(path) for name, path in paths.items()}
        labels = loaded["TSConv"][1]
        if not all(np.array_equal(value[1], labels) for value in loaded.values()):
            raise ValueError(f"candidate order mismatch for subject {subject}")
        available.append(subject)
        for name, (scores, _) in loaded.items():
            by_member[name].append(row_z(scores))

    if not available:
        raise SystemExit("no complete common subjects")
    scores = {name: np.stack(parts) for name, parts in by_member.items()}
    solo = {name: top1(value) for name, value in scores.items()}
    print(f"subjects={available}")
    for name, value in solo.items():
        print(f"solo {name:16s} {value:.2f}")

    for size in (2, 3, 4):
        print(f"k={size}")
        rows = []
        for names in combinations(MEMBERS, size):
            fused = sum(scores[name] for name in names) / size
            accuracy = top1(fused)
            gain = accuracy - max(solo[name] for name in names)
            rows.append((accuracy, gain, names))
        for accuracy, gain, names in sorted(rows, reverse=True):
            print(f"  {' + '.join(names):56s} top1={accuracy:.2f} gain={gain:+.2f}")

    for other in ("ATM", "TSConv", "Squeezeformer"):
        corr = np.corrcoef(
            scores["MultiScaleMixer"].ravel(), scores[other].ravel()
        )[0, 1]
        print(f"corr MultiScaleMixer/{other}={corr:.4f}")


if __name__ == "__main__":
    main()
