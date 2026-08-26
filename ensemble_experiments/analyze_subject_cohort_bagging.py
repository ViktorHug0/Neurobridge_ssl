"""Aggregate three disjoint source-cohort TSConv models per held-out subject."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COHORT_SEED = 20260826


def cohorts_for_target(target: int) -> list[list[int]]:
    sources = np.asarray([subject for subject in range(1, 11) if subject != target])
    shuffled = np.random.default_rng(COHORT_SEED + target).permutation(sources)
    return [sorted(group.tolist()) for group in np.split(shuffled, 3)]


def groups_for_target(target: int, group_mode: str) -> list[list[int]]:
    triplets = cohorts_for_target(target)
    if group_mode == "triplet":
        return triplets
    if group_mode == "overlap6":
        sources = set(range(1, 11)) - {target}
        return [sorted(sources - set(excluded)) for excluded in triplets]
    raise ValueError(f"unsupported group mode: {group_mode}")


def row_z(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), eps
    )


def cosine_scores(eeg: np.ndarray, image: np.ndarray) -> np.ndarray:
    eeg = eeg / np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-8)
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    return (eeg @ image.T).astype(np.float32)


def accuracies(scores: np.ndarray) -> tuple[float, float]:
    targets = np.arange(scores.shape[0])
    top1 = float((scores.argmax(axis=1) == targets).mean() * 100.0)
    top5_indices = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    top5 = float((top5_indices == targets[:, None]).any(axis=1).mean() * 100.0)
    return top1, top5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="results/things_eeg/subject_cohort_bagging/testselected_triplets",
    )
    parser.add_argument("--print-group", type=int, nargs=2, metavar=("TARGET", "COHORT"))
    parser.add_argument(
        "--group-mode", choices=["triplet", "overlap6"], default="triplet"
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.print_group is not None:
        target, cohort = args.print_group
        print(" ".join(map(str, groups_for_target(target, args.group_mode)[cohort])))
        return

    root = Path(args.result_root)
    manifest = {
        f"sub-{target:02d}": groups_for_target(target, args.group_mode)
        for target in range(1, 11)
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "cohort_manifest.json").write_text(
        json.dumps(
            {"seed": COHORT_SEED, "group_mode": args.group_mode, "cohorts": manifest},
            indent=2,
        )
        + "\n"
    )

    missing = []
    rows = []
    for target in range(1, 11):
        score_rows = []
        member_top1 = []
        member_top5 = []
        reference_labels = None
        for cohort in range(3):
            dump = root / f"target-{target:02d}" / f"cohort-{cohort}" / "embeddings.npz"
            if not dump.exists():
                missing.append(str(dump))
                continue
            data = np.load(dump)
            labels = np.stack((data["object"], data["image_idx"]), axis=1)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise RuntimeError(f"candidate ordering mismatch in {dump}")
            scores = cosine_scores(data["eeg"], data["image"])
            top1, top5 = accuracies(scores)
            member_top1.append(top1)
            member_top5.append(top5)
            score_rows.append(row_z(scores))
        if len(score_rows) != 3:
            continue
        score_rows = np.stack(score_rows)
        fused = score_rows.mean(axis=0)
        ensemble_top1, ensemble_top5 = accuracies(fused)
        correlations = np.corrcoef(score_rows.reshape(3, -1))
        mean_correlation = float(correlations[np.triu_indices(3, k=1)].mean())
        rows.append(
            {
                "target": target,
                "cohort_0_subjects": "-".join(
                    map(str, groups_for_target(target, args.group_mode)[0])
                ),
                "cohort_1_subjects": "-".join(
                    map(str, groups_for_target(target, args.group_mode)[1])
                ),
                "cohort_2_subjects": "-".join(
                    map(str, groups_for_target(target, args.group_mode)[2])
                ),
                "cohort_0_top1": member_top1[0],
                "cohort_1_top1": member_top1[1],
                "cohort_2_top1": member_top1[2],
                "mean_member_top1": float(np.mean(member_top1)),
                "best_member_top1": float(np.max(member_top1)),
                "ensemble_top1": ensemble_top1,
                "ensemble_top5": ensemble_top5,
                "gain_over_mean": ensemble_top1 - float(np.mean(member_top1)),
                "gain_over_best": ensemble_top1 - float(np.max(member_top1)),
                "mean_pair_score_correlation": mean_correlation,
            }
        )

    if missing and args.require_complete:
        raise SystemExit(f"waiting for {len(missing)} cohort dumps")
    if not rows:
        print(f"complete targets=0/10; missing dumps={len(missing)}")
        return
    with (root / "cohort_ensemble_partial.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete targets={len(rows)}/10; missing dumps={len(missing)}")
    if len(rows) == 10:
        average = {
            "target": "Average",
            **{
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
                if key
                not in {
                    "target",
                    "cohort_0_subjects",
                    "cohort_1_subjects",
                    "cohort_2_subjects",
                }
            },
        }
        average.update(
            {
                "cohort_0_subjects": "",
                "cohort_1_subjects": "",
                "cohort_2_subjects": "",
            }
        )
        final_rows = rows + [average]
        with (root / "cohort_ensemble.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(final_rows)
        print(json.dumps(average, indent=2))


if __name__ == "__main__":
    main()
