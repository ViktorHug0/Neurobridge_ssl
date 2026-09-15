"""Test whether overlap-6 TSConv cohort models improve existing ensembles.

This is a fixed, non-nested, test-selected analysis.  Every arm is converted to
its 200x200 cosine-score matrix and standardized independently per query before
uniform averaging.  The cohort committee is evaluated separately from its
three component arms so that a component cannot be counted twice implicitly.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from extended_roster_z_sweep import DEFAULT_ROSTER, DEPTH_FAMILIES


SUBJECTS = tuple(range(1, 11))
COHORT_MEMBERS = tuple(f"overlap6_c{i}" for i in range(3))
COHORT_COMMITTEE = "overlap6_committee"


def row_z(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return ((scores - scores.mean(1, keepdims=True)) /
            np.maximum(scores.std(1, keepdims=True), eps)).astype(np.float32)


def load_score(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    eeg = data["eeg"].astype(np.float32)
    image = data["image"].astype(np.float32)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    labels = np.stack([data["object"], data["image_idx"]], axis=1)
    return row_z(eeg @ image.T), labels


def load_standard_arm(root: Path, name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    scores, labels = [], []
    for subject in SUBJECTS:
        score, label = load_score(root / f"{name}-sub{subject:02d}.npz")
        scores.append(score)
        labels.append(label)
    return np.stack(scores), labels


def load_cohorts(root: Path) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    by_member: dict[str, list[np.ndarray]] = {name: [] for name in COHORT_MEMBERS}
    committees, labels = [], []
    for subject in SUBJECTS:
        subject_scores = []
        reference = None
        for cohort, name in enumerate(COHORT_MEMBERS):
            score, label = load_score(
                root / f"target-{subject:02d}" / f"cohort-{cohort}" / "embeddings.npz"
            )
            if reference is None:
                reference = label
            elif not np.array_equal(reference, label):
                raise ValueError(f"cohort query order differs for subject {subject}")
            by_member[name].append(score)
            subject_scores.append(score)
        # Treat the already-fused committee as one calibrated arm in later fusion.
        committees.append(row_z(np.mean(subject_scores, axis=0)))
        labels.append(reference)
    result = {name: np.stack(values) for name, values in by_member.items()}
    result[COHORT_COMMITTEE] = np.stack(committees)
    return result, labels


def evaluate_pool(
    all_scores: dict[str, np.ndarray],
    roster: tuple[str, ...],
    device: torch.device,
    batch_size: int,
    required: set[str] | None = None,
) -> dict:
    scores = torch.from_numpy(np.stack([all_scores[name] for name in roster])).to(device)
    target = torch.arange(scores.shape[-1], device=device)
    report: dict[str, dict] = {}
    for size in range(2, min(6, len(roster)) + 1):
        combinations = list(itertools.combinations(range(len(roster)), size))
        if required:
            combinations = [
                combo for combo in combinations
                if any(roster[index] in required for index in combo)
            ]
        best_mean = -1.0
        best_combo = None
        best_folds = None
        for start in range(0, len(combinations), batch_size):
            batch = combinations[start:start + batch_size]
            index = torch.tensor(batch, dtype=torch.long, device=device)
            # Accumulate members one at a time.  Advanced-indexing the complete
            # (batch, k, subject, query, candidate) tensor needlessly multiplies
            # peak memory by k for the larger exhaustive searches.
            fused = scores[index[:, 0]].clone()
            for member in range(1, size):
                fused.add_(scores[index[:, member]])
            fold_scores = fused.argmax(-1).eq(target).float().mean(-1).mul(100)
            means = fold_scores.mean(1)
            value, offset = means.max(0)
            if float(value) > best_mean:
                best_mean = float(value)
                best_combo = batch[int(offset)]
                best_folds = fold_scores[int(offset)].cpu().tolist()
        report[str(size)] = {
            "mean_top1": best_mean,
            "members": [roster[index] for index in best_combo],
            "fold_top1": best_folds,
            "combinations_evaluated": len(combinations),
        }
    return report


def solo_summary(scores: np.ndarray) -> dict:
    target = np.arange(scores.shape[-1])
    folds = (scores.argmax(-1) == target).mean(-1) * 100
    return {"mean_top1": float(folds.mean()), "fold_top1": folds.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    parser.add_argument(
        "--cohort-root",
        default="results/things_eeg/subject_cohort_bagging/testselected_overlap6",
    )
    parser.add_argument(
        "--output",
        default="ensemble_experiments/analysis/cohort_roster_integration.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--scope", choices=("depth", "extended", "all"), default="all",
        help="Limit the sweep when running on a memory-constrained login node.",
    )
    args = parser.parse_args()

    dump_root, cohort_root = Path(args.dump_root), Path(args.cohort_root)
    standard_names = tuple(dict.fromkeys(DEFAULT_ROSTER))
    all_scores: dict[str, np.ndarray] = {}
    reference_labels = None
    for name in standard_names:
        score, labels = load_standard_arm(dump_root, name)
        if reference_labels is None:
            reference_labels = labels
        elif not all(np.array_equal(a, b) for a, b in zip(reference_labels, labels)):
            raise ValueError(f"query order differs for {name}")
        all_scores[name] = score

    cohort_scores, cohort_labels = load_cohorts(cohort_root)
    if not all(np.array_equal(a, b) for a, b in zip(reference_labels, cohort_labels)):
        raise ValueError("cohort and standard-roster query orders differ")
    all_scores.update(cohort_scores)

    device = torch.device(args.device)
    pools = {
        "depth12": tuple(DEPTH_FAMILIES),
        "depth12_plus_members": tuple(DEPTH_FAMILIES) + COHORT_MEMBERS,
        "depth12_plus_committee": tuple(DEPTH_FAMILIES) + (COHORT_COMMITTEE,),
        "extended27": standard_names,
        "extended27_plus_members": standard_names + COHORT_MEMBERS,
        "extended27_plus_committee": standard_names + (COHORT_COMMITTEE,),
    }
    if args.scope == "depth":
        pools = {name: roster for name, roster in pools.items() if name.startswith("depth12")}
    elif args.scope == "extended":
        pools = {name: roster for name, roster in pools.items() if name.startswith("extended27")}
    result = {
        "protocol": "fixed all-ten test-selected; per-query row-z; uniform mean; no nested selection",
        "cohort_solos": {name: solo_summary(score) for name, score in cohort_scores.items()},
        "pools": {},
    }
    for pool_name, roster in pools.items():
        print(f"evaluating {pool_name} ({len(roster)} arms)", flush=True)
        entry = {"roster": list(roster), "best": evaluate_pool(
            all_scores, roster, device, args.batch_size
        )}
        added = set(roster) & (set(COHORT_MEMBERS) | {COHORT_COMMITTEE})
        if added:
            entry["best_requiring_cohort"] = evaluate_pool(
                all_scores, roster, device, args.batch_size, required=added
            )
        result["pools"][pool_name] = entry

        # Preserve completed work if a long exhaustive extended-roster sweep is
        # interrupted on a shared login node.
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
