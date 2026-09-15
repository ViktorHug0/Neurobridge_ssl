"""Compare TinySqueezeformer with the fixed TinyTSConv/TinyATM reference pair."""
from pathlib import Path

import numpy as np

from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, row_z


REFERENCE = Path("results/things_eeg/tiny_compute_ensemble/testselected_internvit28")
SQF_ROOT = Path("results/things_eeg/tiny_squeezeformer/testselected_internvit28")
MEMBERS = {
    "TSConv": (REFERENCE, "tiny_tsconv_seed3300"),
    "ATM": (REFERENCE, "tiny_atm_seed4300"),
    "Squeezeformer": (SQF_ROOT, "tiny_squeezeformer_seed3300"),
}


def main():
    available = []
    scores = {name: [] for name in MEMBERS}
    labels = []
    for subject in range(1, 11):
        paths = {
            name: root / member / f"sub-{subject:02d}" / "embeddings.npz"
            for name, (root, member) in MEMBERS.items()
        }
        if not all(path.exists() for path in paths.values()):
            continue
        loaded = {name: load_dump(path) for name, path in paths.items()}
        reference_labels = loaded["TSConv"][1]
        if not all(np.array_equal(item[1], reference_labels) for item in loaded.values()):
            raise ValueError(f"label mismatch for subject {subject}")
        available.append(subject)
        labels.append(reference_labels)
        for name, (member_scores, _) in loaded.items():
            scores[name].append(row_z(member_scores))

    if not available:
        raise SystemExit("no complete common subjects")

    stacked = {name: np.stack(parts) for name, parts in scores.items()}
    truth = np.arange(stacked["TSConv"].shape[-1])

    def top1(member_scores):
        return float((member_scores.argmax(axis=-1) == truth).mean() * 100.0)

    print(f"subjects={available}")
    solo = {}
    for name, member_scores in stacked.items():
        solo[name] = top1(member_scores)
        print(f"solo {name:13s} top1={solo[name]:.2f}")
    for left, right in (("ATM", "TSConv"), ("ATM", "Squeezeformer"),
                        ("TSConv", "Squeezeformer")):
        fused = (stacked[left] + stacked[right]) / 2
        pair_top1 = top1(fused)
        corr = np.corrcoef(stacked[left].ravel(), stacked[right].ravel())[0, 1]
        negative_mask = np.ones(stacked[left].shape[-2:], dtype=bool)
        negative_mask[np.arange(len(truth)), truth] = False
        negative_mask = np.broadcast_to(negative_mask, stacked[left].shape)
        negative_corr = np.corrcoef(
            stacked[left][negative_mask], stacked[right][negative_mask]
        )[0, 1]
        gain = pair_top1 - max(solo[left], solo[right])
        print(
            f"pair {left:13s}+{right:13s} top1={pair_top1:.2f} "
            f"gain_over_best={gain:+.2f} corr={corr:.4f} "
            f"negative_corr={negative_corr:.4f}"
        )
    fused = sum(stacked.values()) / len(stacked)
    print(f"triple ATM+TSConv+Squeezeformer top1={top1(fused):.2f}")


if __name__ == "__main__":
    main()
