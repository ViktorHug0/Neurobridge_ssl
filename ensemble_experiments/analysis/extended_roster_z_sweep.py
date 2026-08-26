"""Exhaustive fixed k=2..6 row-z ensemble sweep over the extended model roster.

The protocol is deliberately simple and post-hoc: reconstruct each arm's 200x200
cosine matrix, standardize every query row independently, uniformly sum member
scores, and select the combination with the highest mean over all ten subjects.
No nested member selection or transductive test-time method is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch


DEPTH_FAMILIES = (
    "atm23", "atm25", "atm_iv", "atm31", "atm33", "atm35",
    "iv23", "iv25", "iv28", "iv31", "iv33", "iv35",
)

ARCHITECTURE_ARMS = (
    "sqf28",
    "inductive_v2_ranked_ts_bigru",
    "inductive_v2_multiscale_ts_mixer",
    "temporal_conformer",
    "convnext1d",
    "ortho_mixer_group",
    "ortho_perceiver_group",
    "ortho_convgru_group",
)

PRIOR_FOCUSED_POOL_ADDITIONS = (
    "ge100",
    "tsconv_eva",
    "tsconv_vith",
    "tsconv_bigg",
    "atm_vith",
    "atm_iv_group_e75",
    "iv33_group_e75",
)

DEFAULT_ROSTER = tuple(dict.fromkeys(
    DEPTH_FAMILIES + ARCHITECTURE_ARMS + PRIOR_FOCUSED_POOL_ADDITIONS
))


def load_arm(root: Path, name: str, subjects: tuple[int, ...]):
    matrices = []
    labels = []
    for subject in subjects:
        path = root / f"{name}-sub{subject:02d}.npz"
        data = np.load(path)
        eeg = data["eeg"].astype(np.float32)
        image = data["image"].astype(np.float32)
        eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
        image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
        score = eeg @ image.T
        score = (score - score.mean(1, keepdims=True)) / np.maximum(
            score.std(1, keepdims=True), 1e-6
        )
        matrices.append(score.astype(np.float32))
        labels.append(np.stack([data["object"], data["image_idx"]], axis=1))
    return np.stack(matrices), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="ensemble_experiments/analysis/extended_roster_z_sweep.json")
    parser.add_argument("--roster", nargs="*", default=list(DEFAULT_ROSTER))
    args = parser.parse_args()

    root = Path(args.dump_root)
    subjects = tuple(range(1, 11))
    roster = tuple(dict.fromkeys(args.roster))
    missing = [
        str(root / f"{name}-sub{subject:02d}.npz")
        for name in roster
        for subject in subjects
        if not (root / f"{name}-sub{subject:02d}.npz").exists()
    ]
    if missing:
        raise FileNotFoundError("missing dumps:\n" + "\n".join(missing))

    matrices = []
    reference_labels = None
    for name in roster:
        score, labels = load_arm(root, name, subjects)
        if reference_labels is None:
            reference_labels = labels
        elif not all(np.array_equal(a, b) for a, b in zip(reference_labels, labels)):
            raise ValueError(f"query order differs for {name}")
        matrices.append(score)

    device = torch.device(args.device)
    scores = torch.from_numpy(np.stack(matrices)).to(device)
    target = torch.arange(scores.shape[-1], device=device)
    results = {
        "protocol": "fixed all-ten selection; per-query row-z; uniform arithmetic mean",
        "subjects": list(subjects),
        "roster": list(roster),
        "roster_size": len(roster),
        "best": {},
    }
    print(f"roster={len(roster)} device={device}", flush=True)

    for size in range(2, min(6, len(roster)) + 1):
        combinations = list(itertools.combinations(range(len(roster)), size))
        best_mean = -1.0
        best_fold_scores = None
        best_combo = None
        for start in range(0, len(combinations), args.batch_size):
            batch_combos = combinations[start:start + args.batch_size]
            index = torch.tensor(batch_combos, dtype=torch.long, device=device)
            fused = scores[index[:, 0]].clone()
            for member in range(1, size):
                fused.add_(scores[index[:, member]])
            fold_scores = (
                fused.argmax(dim=-1).eq(target).float().mean(dim=-1).mul_(100.0)
            )
            means = fold_scores.mean(dim=1)
            value, offset = means.max(dim=0)
            value = float(value)
            if value > best_mean:
                best_mean = value
                best_fold_scores = fold_scores[int(offset)].cpu().tolist()
                best_combo = batch_combos[int(offset)]
        names = [roster[index] for index in best_combo]
        results["best"][str(size)] = {
            "mean_top1": best_mean,
            "members": names,
            "fold_top1": best_fold_scores,
            "combinations_evaluated": len(combinations),
        }
        print(
            f"k={size} mean={best_mean:.2f} "
            f"members={' + '.join(names)} folds={best_fold_scores}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
